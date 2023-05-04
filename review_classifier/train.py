#training utilities

#from imblearn.over_sampling import RandomOverSampler
import json
import numpy as np
import pandas as pd

from config.config import logger, DATA_DIR, TEST_SIZE, BATCH_SIZE, CONFIG_DIR
from config.config import logger, NUM_LABELS, DATA_FILENAME

from review_classifier import data_fn, utils
import optuna
import mlflow
from pathlib import Path
from transformers import Trainer, TrainingArguments
from transformers import EvalPrediction  # noqa: E402
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from review_classifier.models import RobertaClassifier
import evaluate
import torch
from argparse import Namespace
from tqdm import tqdm
#TODO: save model checkpoints
#TODO: shuffle data in dataloader with shuffle=True
#TODO: do one final validation prediction on the whole dataset to report instead
#of accuracy
#TODO: learning rate scheduler



def objective_nontrainer(args, df, trial):
    """Objective function for optimization trials. Trial is optuna object from study.optimize."""
    # Parameters to tune
    args.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    args.dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
    args.num_train_epochs= trial.suggest_int("num_train_epochs", 1, 20)
    # Train & evaluate
    artifacts = train_original(args=args, df=df, trial=trial)

    # Set additional attributes
    overall_performance = artifacts["performance"]["overall"]
    logger.info(json.dumps(overall_performance, indent=2))
    trial.set_user_attr("precision", overall_performance["precision"])
    trial.set_user_attr("recall", overall_performance["recall"])
    trial.set_user_attr("f1", overall_performance["f1"])

    return overall_performance["f1"] #return the metric we want to maximize as setup in main.optimize

def train_original(args, df, trial=None):
    utils.set_seeds()

    from transformers import DataCollatorWithPadding
    from transformers import AutoTokenizer
    

    from torch import cuda
    device = 'cuda' if cuda.is_available() else 'cpu'


    model = RobertaClassifier(dropout=args.dropout_rate)
    

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.learning_rate)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    print(tokenizer.is_fast) #should be true

    tokenized_df = create_tokenized_datasets(tokenizer, DATA_FILENAME, "Rating", "Review")
    #need to use data collater to pad with this during each batch with end goal of save memory
    tokenized_df = tokenized_df.remove_columns(["Review", "Rating", '__index_level_0__'])
    #print(tokenized_df['train'].__getitem__(10))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(tokenized_df['train'], collate_fn=data_collator, batch_size=args.batch_size)
    validate_loader = DataLoader(tokenized_df['validate'], collate_fn=data_collator, batch_size=args.batch_size)

    # Training of the model.
    for epoch in range(args.num_train_epochs):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        model.train()
        for batch_idx, data in enumerate(tqdm(train_loader)):
            
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            #token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['labels'].to(device, dtype = torch.long)

            
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)

            if batch_idx%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples 
                logger.info(f"Training Loss per 5000 steps: {loss_step}")
                logger.info(f"Training Accuracy per 5000 steps: {accu_step}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        eval_nb_tr_examples = 0
        val_loss = 0
 
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(validate_loader)):
                ids = data['input_ids'].to(device, dtype = torch.long)
                mask = data['attention_mask'].to(device, dtype = torch.long)
                #token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['labels'].to(device, dtype = torch.long)
                outputs = model(ids, mask)
                loss = loss_function(outputs, targets)
                val_loss += loss.item()
                # Get the index of the max log-probability.
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.view_as(pred)).sum().item()
                eval_nb_tr_examples+=targets.size(0)
        if not epoch%10:
           logger.info(
                f"Epoch: {epoch:02d} | "
                f"train_loss: {tr_loss:.5f}, "
                f"val_loss: {val_loss:.5f}")

        accuracy = correct / eval_nb_tr_examples
        if trial:
            trial.report(accuracy, epoch)
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if not trial:
            mlflow.log_metrics({"train_loss": tr_loss, "val_loss": val_loss}, step=epoch)

        # Handle pruning based on the intermediate value.
    #TODO: can update this to for the last epoch have these metrics calcualted. Or use the test set after all the epochs to do one prediction and return as overall
    accuracy = {'overall': {
        'accuracy': accuracy, 
        'precision': None,
        'recall': None,
        'f1': None}}

    return {
        "args": args,
        "tokenizer": tokenizer,
        "model": model,
        "performance": accuracy,
    }


def hyperparam_search_train(trial):
    utils.set_seeds()
    from torch import cuda
    device = 'cuda' if cuda.is_available() else 'cpu'
    args_fp = Path(CONFIG_DIR, "args.json")
    args = Namespace(**utils.load_dict(filepath=args_fp))

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
    num_train_epochs= trial.suggest_int("num_train_epochs", 1, 20)
    model = RobertaClassifier(dropout=dropout_rate)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=learning_rate)

    tokenized_df = create_tokenized_datasets(tokenizer, DATA_FILENAME, "Rating", "Review")
    train_loader = DataLoader(tokenized_df['train'], **args.train_params)
    validate_loader = DataLoader(tokenized_df['validate'], **args.validate_params)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training of the model.
    for epoch in range(num_train_epochs):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        model.train()
        for batch_idx, data in enumerate(train_loader):
            
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            #token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['labels'].to(device, dtype = torch.long)

            
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)

            if batch_idx%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples 
                print(f"Training Loss per 5000 steps: {loss_step}")
                print(f"Training Accuracy per 5000 steps: {accu_step}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        eval_nb_tr_examples = 0
        val_loss = 0
 
        with torch.no_grad():
            for batch_idx, data in enumerate(validate_loader):
                ids = data['input_ids'].to(device, dtype = torch.long)
                mask = data['attention_mask'].to(device, dtype = torch.long)
                #token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['labels'].to(device, dtype = torch.long)
                outputs = model(ids, mask)
                loss = loss_function(outputs, targets)
                val_loss += loss.item()
                # Get the index of the max log-probability.
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.view_as(pred)).sum().item()
                eval_nb_tr_examples+=targets.size(0)
        mlflow.log_metrics({"train_loss": tr_loss, "val_loss": val_loss}, step=epoch)
        if not epoch%10:
           logger.info(
                f"Epoch: {epoch:02d} | "
                f"train_loss: {tr_loss:.5f}, "
                f"val_loss: {val_loss:.5f}")

        accuracy = correct / eval_nb_tr_examples

        trial.report(accuracy, epoch)
        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def train(args, df, optimized_params=None):
    """Train model on data."""
    #TODO: save best run
    utils.set_seeds()
    from torch import cuda
    device = 'cuda' if cuda.is_available() else 'cpu'

    learning_rate = optimized_params["learning_rate"]
    dropout_rate = optimized_params["dropout_rate"]
    num_train_epochs= optimized_params["num_train_epochs"]
    model = RobertaClassifier(dropout=dropout_rate)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=learning_rate)

    tokenized_df = create_tokenized_datasets(tokenizer, DATA_FILENAME, "Rating", "Review")
    train_loader = DataLoader(tokenized_df['train'], **args.train_params)
    validate_loader = DataLoader(tokenized_df['validate'], **args.validate_params)

    # Training of the model.
    for epoch in range(num_train_epochs):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        model.train()
        for batch_idx, data in enumerate(train_loader):
            
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            #token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['labels'].to(device, dtype = torch.long)

            
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)

            if batch_idx%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples 
                print(f"Training Loss per 5000 steps: {loss_step}")
                print(f"Training Accuracy per 5000 steps: {accu_step}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        eval_nb_tr_examples = 0
        val_loss = 0
 
        with torch.no_grad():
            for batch_idx, data in enumerate(validate_loader):
                ids = data['input_ids'].to(device, dtype = torch.long)
                mask = data['attention_mask'].to(device, dtype = torch.long)
                #token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['labels'].to(device, dtype = torch.long)
                outputs = model(ids, mask)
                loss = loss_function(outputs, targets)
                val_loss += loss.item()
                # Get the index of the max log-probability.
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.view_as(pred)).sum().item()
                eval_nb_tr_examples+=targets.size(0)
        mlflow.log_metrics({"train_loss": tr_loss, "val_loss": val_loss}, step=epoch)
        if not epoch%10:
           logger.info(
                f"Epoch: {epoch:02d} | "
                f"train_loss: {tr_loss:.5f}, "
                f"val_loss: {val_loss:.5f}")
        accuracy = correct / eval_nb_tr_examples
    #TODO: save model here to return None or both return dict and save model
    return {
        "args": args,
        "tokenizer": tokenizer,
        "model": model,
        "performance": accuracy,
    }
   

    

def create_tokenized_datasets(tokenizer, datafile_name, label_col_name, text_col_name):
    """returns a huggingface DatasetDict object with train, validate, and test columns. \
    There should also be two columns in each dataset we are interested in "text" and "label" \
    Trainer object will automatically move things to a tensor as needed for us. """
    def tokenize_function(example):
        #old handling: tokenized_outputs = tokenizer(text, return_tensors="pt")
        tokens = tokenizer(example[text_col_name], truncation=True, padding=False)
        tokens['labels'] = labels.str2int(example[label_col_name])
        return tokens

    #load dataset class object
    df, labels = pytorch_dataset(datafile_name, label_col_name)
    #transform dataset label

    #tokenize dataset (doing it this way so the results get pushed back as new columns in Datasets format stored on Disk instead of returning dictionary stored in RAM)
    tokenized_datasets = df.map(tokenize_function, batched=True)
    return tokenized_datasets


    
def pytorch_dataset(filename, label_col_name):
    from datasets import Dataset, DatasetDict, ClassLabel
    train, validate, test, labels_set = data_fn.optimization_read_split_data(
        path= Path(DATA_DIR, filename),
        test_size=TEST_SIZE,
        label_col_name=label_col_name,
    )
    train = Dataset.from_pandas(train)
    validate = Dataset.from_pandas(validate)
    test = Dataset.from_pandas(test)
    dataset = DatasetDict({
        "train": train,
        "validate": validate,
        "test": test})
    labels = ClassLabel(names = list(labels_set))
    return dataset, labels

def optimization_etl_data(filename, label_col_name):
    x_train, x_val, x_test, y_train, y_val, y_test = data_fn.optimization_read_split_data(
        path= Path(DATA_DIR, filename),
        test_size=TEST_SIZE,
        label_col_name=label_col_name,
    )
    
    train_dataloader = data_fn.dataloader(x_train, y_train, BATCH_SIZE)
    val_dataloader = data_fn.dataloader(x_val, y_val, BATCH_SIZE)
    test_dataloader = data_fn.dataloader(x_test, y_test, BATCH_SIZE)
    return train_dataloader, val_dataloader, test_dataloader

def training_etl_data(filename, label_col_name):
    x_train, x_test, y_train, y_test = data_fn.training_read_split_data(
        path= Path(DATA_DIR, filename),
        test_size=TEST_SIZE,
        label_col_name=label_col_name,
    )

    train_dataloader = data_fn.dataloader(x_train, y_train, BATCH_SIZE)
    test_dataloader = data_fn.dataloader(x_test, y_test, BATCH_SIZE)
    return train_dataloader, test_dataloader


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


def compute_metrics2(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)     
    return metric.compute(predictions=predictions, references=labels)

def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 20),
    }

def my_objective(metrics):
    return metrics['eval_loss']
