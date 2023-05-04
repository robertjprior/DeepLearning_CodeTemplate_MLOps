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
import mlflow
from numpyencoder import NumpyEncoder
import optuna
from optuna.integration.mlflow import MLflowCallback

#roberta sentence classification https://colab.research.google.com/github/DhavalTaunk08/NLP_scripts/blob/master/sentiment_analysis_using_roberta.ipynb#scrollTo=bFiNcy16JLwt

def train_model(args_fp, experiment_name, run_name):  # noqa: E501
    """
    Train a model given arguments. wrapper for train function from train.py \
        such that this will train a final model after optimization has been done \
        and use mlflow to log metrics. MLFlow logs from optimize but I believe only \
        some very high level metrics. """
    # Train
    df= None #done inside train function now
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")
        artifacts = train.train_original(df=df, args=args)
        performance = artifacts["performance"]
        logger.info(json.dumps(performance, indent=2))

        # Log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

    # Save to config
    open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
    utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))
    """There's a lot more happening inside our train_model() function but it's necessary in order to store all the metrics, parameters and artifacts. We're also going to update the train() function inside train.py so that the intermediate metrics are captured:"""

   
#optuna in pytorch: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
#optimize params setup in train.py under objective function
from review_classifier import train
def optimize(args_fp, experiment_name, run_name, num_trials):
    """Optimize hyperparameters."""
    args = Namespace(**utils.load_dict(filepath=args_fp))
    
    df=None

    # Optimize
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name="optimization", direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(), metric_name="f1") #passing MLFlow tracking uri to optuna MLflowCallback to pass info 
    #https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.MLflowCallback.html#optuna.integration.MLflowCallback
    study.optimize(
        lambda trial: train.objective_nontrainer(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback]) #replaces manually adding trail with custom params setups etc https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/007_optuna_callback.html

    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    utils.save_dict({**args.__dict__, **study.best_trial.params}, args_fp, cls=NumpyEncoder)
    logger.info(f"\nBest value (f1): {study.best_trial.value}")
    logger.info(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")
    


from review_classifier.train import hyperparam_search_train as objective
def hyperparam_search(args_fp, experiment_name, run_name, num_trials):
    """old hyperparam search being built to do train and optimize with just separate functions"""
    #https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
    args = Namespace(**utils.load_dict(filepath=args_fp))

    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        study = optuna.create_study(direction="maximize")

        study.optimize(objective, n_trials=num_trials, timeout=600)
        pruned_trials = None #study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = None #study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        logger.info("Study statistics: ")
        logger.info("  Number of finished trials: ", len(study.trials))
        logger.info("  Number of pruned trials: ", len(pruned_trials))
        logger.info("  Number of complete trials: ", len(complete_trials))

        logger.info("Best trial:")
        trial = study.best_trial

        logger.info("  Value: ", trial.value)

        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info("    {}: {}".format(key, value))
        best_run = study.best_trial
        mlflow.log_params(vars(best_run))
        logger.info(json.dumps(best_run, indent=2))
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        utils.save_dict(best_run, Path(config.CONFIG_DIR, "performance.json"))
        utils.save_dict(best_run.params, Path(config.CONFIG_DIR, "best_hyperparameters.json"))
