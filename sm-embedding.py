from sagemaker import Session
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.huggingface.model import HuggingFacePredictor
from sagemaker.huggingface import get_huggingface_llm_image_uri
import pandas as pd 
import os 

# parameters 
role = os.environ["sm_execution_role"]
instance_type="ml.m5.xlarge"

# sm session 
session = Session()

# hf hub 
hub = {
  'HF_MODEL_ID': 'sentence-transformers/all-MiniLM-L6-v2',
  'HF_TASK': 'feature-extraction'
}

# hf model 
hf_model = HuggingFaceModel(
    role=role,
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39",
    env=hub
)

# deploy model 
# hf_model.deploy(
#     role=role,
#     initial_instance_count=1, 
#     instance_type=instance_type,
# )


# predictor 
predictor = HuggingFacePredictor(
    endpoint_name="huggingface-pytorch-inference-2023-09-19-09-20-23-800",
    sagemaker_session=session
)

response = predictor.predict(
    {
        "inputs": "this is a tree"
    }
)

# save embedding to csv 
pd.DataFrame(response[0]).to_csv("embedding.csv", header=None, sep=',')

