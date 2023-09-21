---
title: deploy huggingface model on sagemaker
author: haimtran
description:
date: 21/09/2023
---

## Introduction

[GitHub]() this note summarize different ways to deploy a HuggingFace model on Amazon SageMaker

- SageMaker Model
- HuggingFaceModel
- JumpStartModel

## SageMaker Model

A model in SageMaker consist of image uri and model data.

```py
from sagemaker import Session
from sagemaker.model import Model
from sagemaker import image_uris
from sagemaker import model_uris
import json
```

Let retrieve an image uri

```py
model_id = "huggingface-text2text-flan-t5-xxl"
model_version = "*"
```

and

```py
image_uri = image_uris.retrieve(
  region=None,
  framework=None,
  image_scope="inference",
  model_id=model_id,
  model_version=model_version,
  instance_type="ml.g5.12xlarge"
)
```

then retrieve model data

```py
model_uri = model_uris.retrieve(
  model_id=model_id,
  model_version=model_version,
  model_scope="inference"
)
```

Now we can create a Model object

```py
model_inference = Model(
  image_uri=deploy_image_uri,
  model_data=model_uri,
  role=role,
  predictor_cls=None,
  name=endpoint_name,
  env={"SAGEMAKER_MODEL_SERVER_WORKERS": "1", "TS_DEFAULT_WORKERS_PER_MODEL":"1"}
)
```

Let deploy the model to create an endpoint. Check the last section to invoke the endpoint.

```py
model_inference.deploy(
  initial_instance_count=1,
  instance_type="ml.g5.12xlarge",
  endpoint_name=endpoint_name
)
```

## HuggingFaceModel

Another method is using HuggingFaceModel object. In this method, the model data can be loaded after the endpoint is created

```py
hub = {
  'HF_MODEL_ID': 'sentence-transformers/all-MiniLM-L6-v2',
  'HF_TASK': 'feature-extraction'
}
```

let create a HF model object

```py
hf_model = HuggingFaceModel(
  role=None,
  # modeal data could be loaded after endpoint created
  # model_data=model_data,
  transformers_version="4.26",
  pytorch_version="1.13",
  py_version="py39",
  env=hub
)
```

Then we can call deploy method on the HuggingFaceModel to create an endpoint

## HuggingFace LLM DLC

The third method is using get_huggingface_llm_image_uri

```py
# get image uri from aws ecr
# same hugging face llm dlc for different models
llm_image_uri = get_huggingface_llm_image_uri(
  session=session,
  version="0.9.3",
  backend="huggingface"
)
```

let specify a model

```py
hf_model_id = "OpenAssistant/pythia-12b-sft-v8-7k-steps" # model id from huggingface.co/models
use_quantization = False # wether to use quantization or not
instance_type = "ml.g5.12xlarge" # instance type to use for deployment
number_of_gpu = 4 # number of gpus to use for inference and tensor parallelism
health_check_timeout = 300 # Increase the timeout for the health check to 5 minutes for downloading the model
```

Now create a HuggingFaceModel

```py
llm_model = HuggingFaceModel(
    role=None,
    image_uri=llm_image_uri,
    env={
    'HF_MODEL_ID': hf_model_id,
    'HF_MODEL_QUANTIZE': json.dumps(use_quantization),
    'SM_NUM_GPUS': json.dumps(number_of_gpu)
    }
)
```

## Gradio

The best way would be using SageMaker JumpStartModel

```py
from sagemaker.jumpstart.model import JumpStartModel
```

Let deploy a meta-textgenerate-llama-2-7b-f model

```py
model_id, model_version = "meta-textgeneration-llama-2-7b-f", "*"

model = JumpStartModel(
    model_id=model_id,
    model_version=model_version,
    role=role
)

# it takes about 10 minutes
predictor = model.deploy()
```

After the enpoint is deployed, let use gradio to build a simple chatbot

```py
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("## Chat with Amazon SageMaker")
    with gr.Column():
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column():
                message = gr.Textbox(label="Chat Message Box", placeholder="Chat Message Box", show_label=False)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    clear = gr.Button("Clear")

    def respond(message, chat_history):
        # convert chat history to prompt
        converted_chat_history = ""
        #
        prompt = [[{"role": "user", "content": message}]]
        # send request to endpoint
        llm_response = predictor.predict({"inputs": prompt, "parameters": parameters}, custom_attributes='accept_eula=true')
        # remove prompt from response
        parsed_response = llm_response[0]['generation']['content']
        # parsed_response = llm_response[0]["generated_text"][len(prompt):]
        chat_history.append((message, parsed_response))
        return "", chat_history

    submit.click(respond, [message, chatbot], [message, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(share=True)
```

## Invoke Endpoint

To invoke an endoint, we can create a predictor

```py
from sagemaker.huggingface.model import HuggingFacePredictor
from sagemaker import Session
import numpy as np
```

Define a model

```py
model_id = "huggingface-text2text-flan-t5-xxl"
model_version = "*"
endpoint_name = "hugging_face_llm_demo"
endpoint_name_deployed = "huggingface-pytorch-inference-2023-09-19-05-00-49-267"
```

Create a HuggingFacePredictor

```py
predictor = HuggingFacePredictor(
  endpoint_name=endpoint_name_deployed,
  sagemaker_session=session
)
```

then invoke endpoint

```py
response = predictor.predict({
   "inputs": "Today is a sunny day and I'll get some ice cream."
})
```

## Reference

- [SageMaker JumpStartModel](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart-foundation-models/llama-2-text-completion.ipynb)

- [sagemaker-huggingface-inference-toolkit](https://github.com/aws/sagemaker-huggingface-inference-toolkit)

- [Deploy Falcon 180B on Amazon SageMaker](https://www.philschmid.de/sagemaker-falcon-180b)

- [Deploy models to Amazon SageMaker](https://huggingface.co/docs/sagemaker/inference)
