# Custom Handler Tutorial for TorchServe

In this tutorial, we'll walk you through creating a custom handler for TorchServe using Python. We'll use a sample handler code as a reference, explaining each part and providing a step-by-step guide. By the end of this tutorial, you should be able to create and deploy your custom handler for TorchServe.

## Prerequisites

Before you begin, make sure you have the following:

- Python installed on your system.
- PyTorch and TorchServe installed.
- A trained model in PyTorch format that you want to deploy.

## 1. Import Necessary Libraries

First, let's import the required libraries:

```python
import torch
from ts.torch_handler.base_handler import BaseHandler
from torchvision.io import read_image
from torchvision.models import ResNet50_Weights
import json
import base64
import logging
import os
from PIL import Image, ImageOps
import io
import numpy as np
from model import net  # Import your model here
```
Here, we import various libraries for handling image data, loading the model, and performing image transformations.


## 2. Create a Custom Handler Class
Next, we define our custom handler class, which should inherit from BaseHandler:

```python
class CustomHandler(BaseHandler):
```


## 3. Initialize the Handler
In the __init__ method, we set up some initial variables:
```python
def __init__(self):
    super().__init__()
    self._context = None
    self.initialized = False
    self.explain = False
    self.target = 0
```


## 4. Initialize the Model
In the initialize method, we load and initialize the model:
```python
def initialize(self, context):
    self._context = context
    self.initialized = True

    self.manifest = context.manifest
    properties = context.system_properties
    model_dir = properties.get("model_dir")
    serialized_file = self.manifest['model']['serializedFile']
    model_pt_path = os.path.join(model_dir, serialized_file)

    if not os.path.isfile(model_pt_path):
        raise RuntimeError("Missing the checkpoint model file")

    self.model = net  # Load your model here
    self.model.load_state_dict(torch.load(model_pt_path))
    self.model.eval()

```
In this step, we initialize the model by loading its weights from the provided checkpoint file.

## 5. Load and Process the Image
The _load_image method is responsible for loading and processing the input image:
```python
def _load_image(self, data):
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    return np.array(image)
```
This method takes input data, which is expected to be an image, and returns a NumPy array.

## 6. Handle Prediction
In the handle method, we handle the prediction request:
```python
def handle(self, data, context):
```
This method is invoked by TorchServe for prediction requests.

Here's a breakdown of the steps within the handle method:

- Load and preprocess the input image.
- Perform inference using the loaded model.
- Process the model's prediction output.
- Return the prediction result as a JSON object.
The code within this method processes the image, feeds it to the model, and returns the prediction result.











