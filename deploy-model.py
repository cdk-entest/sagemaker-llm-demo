# haimtran 19/09/2023 
# deploy Hugging Face LLM models on SageMaker 

from sagemaker import Session
from sagemaker import image_uris
from sagemaker import model_uris
from sagemaker.model import Model
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.huggingface.model import HuggingFacePredictor
import os 

# parameter 
model_id = "huggingface-text2text-flan-t5-xxl"
model_version = "*"
role = os.environ["sm_execution_role"]
endpoint_name = "hugging_face_llm_demo"

# sagemaker session  
session = Session()

# default bucket 
bucket = session.default_bucket()

# get image uri 
deploy_image_uri = image_uris.retrieve(
  region=None, 
  framework=None, 
  image_scope="inference",
  model_id=model_id,
  model_version=model_version,
  instance_type="ml.g5.12xlarge"
) 

# get model artifact from s3 
model_uri = model_uris.retrieve(
  model_id=model_id,
  model_version=model_version,
  model_scope="inference"
)

# deploy the model 
model_inference = Model(
  image_uri=deploy_image_uri, 
  model_data=model_uri,
  role=role, 
  predictor_cls=None, 
  name=endpoint_name, 
  env={"SAGEMAKER_MODEL_SERVER_WORKERS": "1", "TS_DEFAULT_WORKERS_PER_MODEL": "1"}
)

# hugging face model 
hugging_face_model = HuggingFaceModel(
  image_uri=deploy_image_uri,
  model_data=model_uri, 
  role=role,
  endpoint_name=endpoint_name,
  predictor_cls=None
)
# 
model_inference.deploy(
  initial_instance_count=1, 
  instance_type="ml.g5.12xlarge",
  endpoint_name=endpoint_name
)


# Hub Model configuration. https://huggingface.co/models
hub = {
	'HF_MODEL_ID':'nreimers/MiniLM-L6-H384-uncased',
	'HF_TASK':'feature-extraction'
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
	transformers_version='4.26.0',
	pytorch_version='1.13.1',
	py_version='py39',
	env=hub,
	role=role, 
)

# deploy model to SageMaker Inference
# predictor = huggingface_model.deploy(
# 	initial_instance_count=1, # number of instances
# 	instance_type='ml.m5.xlarge' # ec2 instance type
# )



# predictor 
# predictor = HuggingFacePredictor(
#   endpoint_name="huggingface-pytorch-inference-2023-09-19-05-00-49-267",
#   sagemaker_session=session,
# )

# response = predictor.predict(
#   {
#     "inputs": "Today is a sunny day and I'll get some ice cream.",
#   }
# )

# print(response)


# predictor.predict({
# 	"inputs": "Today is a sunny day and I'll get some ice cream.",
# })

print(deploy_image_uri)
print(model_uri)
# print(model_inference)
