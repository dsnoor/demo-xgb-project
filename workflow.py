from kfp import dsl
from mlrun import mount_v3io

funcs = {}

def init_functions(functions: dict, params=None, secrets=None):
    for f in functions.values():
        f.apply(mount_v3io())

import os
ARTIFACT_PATH = '/User/demo-xgb-project/artifacts'
os.makedirs(ARTIFACT_PATH, exist_ok=True)

        
@dsl.pipeline(
    name='My XGBoost training pipeline',
    description='Shows how to use mlrun.'
)
def kfpipeline():
    ingest = funcs['get_toy_data'].as_step(
        name='acquire',
        params={'dataset': 'iris'},
        outputs=['data'],
        out_path=ARTIFACT_PATH)

    describe = funcs['describe'].as_step(
        name='describe',
         params={"key": "summary", "label_column": "labels"},
        inputs={"table": ingest.outputs['data']},
        outputs=['data'],
        out_path=ARTIFACT_PATH)
    
    configs = funcs['get_model_config'].as_step(
        name='config',
        params={'config': 'xgboost-conf.json'},
        outputs=['class_params', 'fit_params'],
        out_path=ARTIFACT_PATH)
    
    train = funcs['train_model'].as_step(
        name='train',
        # hyperparams={'eta': eta, 'gamma': gamma},
        # selector='max.accuracy',
        params={'sample': -1, 'label_column': 'labels'}
        inputs={'data_key'    : ingest.outputs['data'],
                'class_params': configs.outputs['class_params'],
                'fit_params'  : configs.outputs['fit_params']},
        outputs=['model'],
        out_path=ARTIFACT_PATH)

    # deploy the trained model using a nuclio real-time function
    deploy = funcs['serving'].deploy_step(models={'iris_v1': train.outputs['model']})
