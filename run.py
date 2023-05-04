from config import config
from review_classifier import main
from pathlib import Path

args_fp = Path(config.CONFIG_DIR, "args.json")
main.optimize(args_fp, experiment_name="optimization", run_name = "baseline", num_trials=20)
#We should see our experiment in our model registry, located at stores/model/:
#this will upload training hyperparams to args so main.train_model will run


#THIS MAY NOT BELONG HERE - OLD generic training run
# from config import config
# from my_package import main
# args_fp = Path(config.CONFIG_DIR, "args.json")
# main.train_model(args_fp)

#new run main model but with experiment tracking now with mlflow for manually trying out different param searches/models
args_fp = Path(config.CONFIG_DIR, "args.json")
main.train_model(args_fp, experiment_name="baselines", run_name="sgd")
"""Our configuration directory should now have a performance.json and a run_id.txt file. We're saving these so we can quickly access this metadata of the latest successful training. If we were considering several models as once, we could manually set the run_id of the run we want to deploy or programmatically identify the best across experiments.


config/
├── args.json         - arguments
├── config.py         - configuration setup
├── performance.json  - performance metrics
└── run_id.txt        - ID of latest successful run

And we should see this specific experiment and run in our model registry:


stores/model/
├── 0/
└── 1/"""

"""Run ID: d91d9760b2e14a5fbbae9f3762f0afaf
Epoch: 00 | train_loss: 0.74266, val_loss: 0.83335
Epoch: 10 | train_loss: 0.21884, val_loss: 0.42853
Epoch: 20 | train_loss: 0.16632, val_loss: 0.39420
Epoch: 30 | train_loss: 0.15108, val_loss: 0.38396
Epoch: 40 | train_loss: 0.14589, val_loss: 0.38089
Epoch: 50 | train_loss: 0.14358, val_loss: 0.37992
Epoch: 60 | train_loss: 0.14084, val_loss: 0.37977
Epoch: 70 | train_loss: 0.14025, val_loss: 0.37828
Epoch: 80 | train_loss: 0.13983, val_loss: 0.37699
Epoch: 90 | train_loss: 0.13841, val_loss: 0.37772
{
  "overall": {
    "precision": 0.9026155077984347,
    "recall": 0.8333333333333334,
    "f1": 0.8497010532479641,
    "num_samples": 144.0
  },
  "class": {
    "computer-vision": {
      "precision": 0.975609756097561,
      "recall": 0.7407407407407407,
      "f1": 0.8421052631578947,
      "num_samples": 54.0
    },
    "mlops": {
      "precision": 0.9090909090909091,
      "recall": 0.8333333333333334,
      "f1": 0.8695652173913043,
      "num_samples": 12.0
    },
    "natural-language-processing": {
      "precision": 0.9807692307692307,
      "recall": 0.8793103448275862,
      "f1": 0.9272727272727272,
      "num_samples": 58.0
    },
    "other": {
      "precision": 0.475,
      "recall": 0.95,
      "f1": 0.6333333333333334,
      "num_samples": 20.0
    }
  },
  "slices": {
    "nlp_cnn": {
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0,
      "num_samples": 1
    },
    "short_text": {
      "precision": 0.8,
      "recall": 0.8,
      "f1": 0.8000000000000002,
      "num_samples": 5
    }
  }
}"""

from my_package.main import predict_tag
text = "Transfer learning with transformers for text classification."
run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
predict_tag(text=text, run_id=run_id)
from my_package.main import predict



