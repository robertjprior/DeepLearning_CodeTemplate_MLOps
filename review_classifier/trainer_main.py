#training/optimization pipelines
import joblib
import tempfile
from review_classifier import data_fn, predict, utils
from config.config import logger, NUM_LABELS, DATA_FILENAME
from review_classifier.train import my_hp_space, my_objective, compute_metrics2, optimization_etl_data, training_etl_data, create_tokenized_datasets
from review_classifier.models import RobertaClassifier
from transformers import Trainer, TrainingArguments
import pandas as pd
from pathlib import Path
from argparse import Namespace
import warnings
from config import config
warnings.filterwarnings("ignore")
import mlflow
#from numpyencoder import NumpyEncoder
import optuna
import json
#from optuna.integration.mlflow import MLflowCallback




"""
#original train_model before optimize
def train_model(args_fp): #This function calls for a train() function inside our train.py script:
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    artifacts = train.train(df=df, args=args)
    performance = artifacts["performance"]
    logger.info(json.dumps(performance, indent=2))"""

def model_init():
    return RobertaClassifier()

def train_model(args_fp, experiment_name, run_name, use_existing_optimization_params= None):  # noqa: E501
    """
    Train a model given arguments. wrapper for train function from train.py \
        such that this will train a final model after optimization has been done \
        and use mlflow to log metrics. MLFlow logs from optimize but I believe only \
        some very high level metrics. """
    from transformers import DataCollatorWithPadding
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    print(tokenizer.is_fast) #should be true
    tokenized_df = create_tokenized_datasets(tokenizer, DATA_FILENAME, "Rating", "Review")
    
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    """The function that is responsible for putting together samples inside a batch is called a collate function. It’s an argument you can pass when you build a DataLoader, the default being a function that will just convert your samples to PyTorch tensors and concatenate them (recursively if your elements are lists, tuples, or dictionaries). This won’t be possible in our case since the inputs we have won’t all be of the same size. We have deliberately postponed the padding, to only apply it as necessary on each batch and avoid having over-long inputs with a lot of padding. This will speed up training by quite a bit, but note that if you’re training on a TPU it can cause problems — TPUs prefer fixed shapes, even when that requires extra padding.
    To do this in practice, we have to define a collate function that will apply the correct amount of padding to the items of the dataset we want to batch together. Fortunately, the 🤗 Transformers library provides us with such a function via DataCollatorWithPadding. It takes a tokenizer when you instantiate it (to know which padding token to use, and whether the model expects padding to be on the left or on the right of the inputs) and will do everything you need:"""
    #check its working
    #batch = data_collator(samples)
    #{k: v.shape for k, v in batch.items()}


    # Load labeled data
    args = Namespace(**utils.load_dict(filepath=args_fp))
    best_optimized_hyperparams = utils.load_dict(Path(config.CONFIG_DIR, "best_hyperparameters.json"))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")

        #TODO: set the seed somewhere so this train test split is the same
        training_args = TrainingArguments(
            output_dir='config',   # output directory
            evaluation_strategy="epoch",     # Evaluation is done at the end of each epoch.
            per_device_eval_batch_size=64,   # batch size for evaluation
            save_total_limit=2,              # limit the total amount of checkpoints. Deletes the older checkpoints.
            **best_optimized_hyperparams    
        )

        trainer = Trainer(
            args=training_args,                  # training arguments, defined above
            train_dataset=tokenized_df['train'],       # training dataset
            eval_dataset=tokenized_df['validate'],
            data_collator=data_collator,
            compute_metrics=compute_metrics2,     # metrics to be computed
            model_init=model_init,                # Instantiate model before training starts
        )

        trainer.train()

    # Train
   


        #artifacts = train.train(df=df, args=args)
        #performance = artifacts["performance"]
        #logger.info(json.dumps(performance, indent=2))

        # Log metrics and parameters
        #performance = artifacts["performance"]
        #mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        #mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        #mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        #mlflow.log_params(vars(artifacts["args"]))

        

    # Save to config
    open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
    #utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))
    """There's a lot more happening inside our train_model() function but it's necessary in order to store all the metrics, parameters and artifacts. We're also going to update the train() function inside train.py so that the intermediate metrics are captured:"""



#optimize params setup in train.py under objective function
def optimize(args, experiment_name, run_name, num_trials):
    """Optimize hyperparameters."""
    from transformers import DataCollatorWithPadding
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    print(tokenizer.is_fast) #should be true
    tokenized_df = create_tokenized_datasets(tokenizer, DATA_FILENAME, "Rating", "Review")
    tokenized_df = tokenized_df.remove_columns(["Review", "Rating", '__index_level_0__'])
    #print(tokenized_df['train'].__getitem__(10))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")
        
        #train_dataloader, val_dataloader, test_dataloader = optimization_etl_data(DATA_FILENAME, "Rating")
        #model = RobertaClassifier()
        training_args = TrainingArguments(
            output_dir='config',    # output directory
            #evaluation_strategy="epoch",     # Evaluation is done at the end of each epoch.
            evaluation_strategy="steps",      # Evaluation is done (and logged) every eval_steps.
            eval_steps=1000,                  # Number of update steps between two evaluations 
            per_device_eval_batch_size=64,    # batch size for evaluation
            save_total_limit=1,               # limit the total amount of checkpoints. Deletes the older checkpoints.
            #remove_unused_columns=False,
        )

        trainer = Trainer(
            args=training_args,                  # training arguments, defined above
            train_dataset=tokenized_df['train'],       # training dataset
            eval_dataset=tokenized_df['validate'],
            data_collator=data_collator,
            compute_metrics=compute_metrics2,     # metrics to be computed
            model_init=model_init,                # Instantiate model before training starts
        )
        best_run = trainer.hyperparameter_search(
            direction="minimize", 
            hp_space=my_hp_space, 
            compute_objective=my_objective, 
            n_trials=num_trials)
        mlflow.log_params(vars(best_run.hyperparameters))
        logger.info(json.dumps(best_run, indent=2))
    # Load labeled data
    open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
    utils.save_dict(best_run, Path(config.CONFIG_DIR, "performance.json"))
    utils.save_dict(best_run.hyperparameters, Path(config.CONFIG_DIR, "best_hyperparameters.json"))
    #logger.info(f"\nBest value (f1): {study.best_trial.value}")
    #logger.info(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")




#from review_classifier import data, predict, train, utils
#KEY
# def predict_tag(text, run_id=None):
#     """Predict tag for text."""
#     if not run_id:
#         run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
#     artifacts = load_artifacts(run_id=run_id)
#     prediction = predict.predict(texts=[text], artifacts=artifacts)
#     logger.info(json.dumps(prediction, indent=2))
#     return prediction

# def load_artifacts(run_id):
#     """Load artifacts for a given run_id."""
#     # Locate specifics artifacts directory
#     experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
#     artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

#     # Load objects from run
#     args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
#     vectorizer = joblib.load(Path(artifacts_dir, "vectorizer.pkl"))
#     label_encoder = data.LabelEncoder.load(fp=Path(artifacts_dir, "label_encoder.json"))
#     model = joblib.load(Path(artifacts_dir, "model.pkl"))
#     performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

#     return {
#         "args": args,
#         "label_encoder": label_encoder,
#         "vectorizer": vectorizer,
#         "model": model,
#         "performance": performance
#     }


# def elt_data():
#     """Extract, load and transform our data assets."""
#     # Extract + Load
#     projects = pd.read_csv(config.PROJECTS_URL)
#     tags = pd.read_csv(config.TAGS_URL)
#     projects.to_csv(Path(config.DATA_DIR, "projects.csv"), index=False)
#     tags.to_csv(Path(config.DATA_DIR, "tags.csv"), index=False)

#     # Transform
#     df = pd.merge(projects, tags, on="id")
#     df = df[df.tag.notnull()]  # drop rows w/ no tag
#     df.to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)

#     logger.info("✅ Saved data!")