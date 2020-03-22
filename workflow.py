from kfp import dsl
from mlrun import mount_v3io

DATASET = "iris"
LABELS  = "labels"
MODEL_KEY = "models"

import os
# path is created automatically if it doesn"t exists
DATA_PATH = "/User/demo-xgb-project/artifacts"
os.makedirs(DATA_PATH, exist_ok=True)

funcs = {}

def init_functions(functions: dict, params=None, secrets=None):
    for f in functions.values():
        f.apply(mount_v3io())

@dsl.pipeline(
    name="My XGBoost training pipeline",
    description="Shows how to use mlrun."
)
def kfpipeline():
    ingest = funcs["get_data"].as_step(
        name="get data",
        params={"dataset": DATASET},
        outputs=[DATASET],
        out_path=DATA_PATH)

    describe = funcs["summary"].as_step(
        name="summary",
        params={"key": "summary", "label_column": LABELS},
        inputs={"table": ingest.outputs[DATASET]},
        outputs=[DATASET])
    
    train = funcs["train"].as_step(
        name="train",
        params={"model_pkg_class" : "sklearn.linear_model.LogisticRegression",
                "model_key"       : MODEL_KEY, 
                "sample"          : -1, 
                "label_column"    : LABELS,
                "test_size"       : 0.10,
                "class_params_updates"  : {"random_state": 1},
                "fit_params_updates"    : {}},
        inputs={"data_key"        : ingest.outputs[DATASET]},
        outputs=[MODEL_KEY, "test_set"])

    test = funcs["test"].as_step(
        name="test",
        params={"label_column": LABELS},
        inputs={"models_dir"  : train.outputs[MODEL_KEY],
                "test_set"    : train.outputs["test_set"]},
        outputs=[MODEL_KEY])

    deploy = funcs["server"].deploy_step(
       project=DATASET, 
       models={f"{DATASET}_v1": train.outputs[MODEL_KEY]})