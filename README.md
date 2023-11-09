# TorchServe Tutorial

This tutorial explains how to serve PyTorch models with TorchServe.

## Prerequisites

- PyTorch  
- TorchServe

## install requirement 
```bash
# Latest release
pip install torchserve torch-model-archiver torch-workflow-archiver

# install requirement 
pip install -r requirement.txt
```

## Steps 

### 1. Create model store directory

```bash
mkdir model_store
```

### 2. Download pre-trained model


```bash
wget https://download.pytorch.org/models/densenet161-8d451a50.pth
```

### 3. Archive model using model archiver

```bash
torch-model-archiver --model-name densenet161 --version 1.0 --model-file model.py --serialized-file densenet161-8d451a50.pth --export-path model_store/ --extra-files index_to_name.json --handler image_classifier
 ```
### 4. Start TorchServe to serve model

```bash
torchserve --start --model-store model_store --models densenet161.mar
```
### 5. Run inference
Send image to API endpoint:
```bash
curl http://127.0.0.1:8080/predictions/densenet161 -T kitten_small.jpg
```
### 6. Stop TorchServe
```bash
torchserve --stop
```

## Client Examples
Python Requests

```python
import requests

url = 'http://127.0.0.1:8080/predictions/densenet161'
files = {'data': open('kitten_small.jpg', 'rb')}

response = requests.post(url, files=files) 
print(response.text)
```

FastAPI

```python
from fastapi import FastAPI, UploadFile, File
import httpx  
import os

app = FastAPI()

@app.post("/predict/{model_name}")  
async def predict(model_name: str, file: UploadFile = File(...)):

  async with httpx.AsyncClient() as client:
    
    with open("temp_image.jpg", "wb") as f: 
      f.write(file.file.read())
      
    response = await client.post(f"http://127.0.0.1:8080/predictions/{model_name}",
                                 files={"data": open("temp_image.jpg", "rb")})
    
    predictions = response.json()
    
    os.remove("temp_image.jpg")
    
    return predictions

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```