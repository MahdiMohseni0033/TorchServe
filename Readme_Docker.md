# TorchServe Tutorial using Docker

In this tutorial, we will walk you through the process of serving PyTorch models using TorchServe in a Docker container.

## Prerequisites
Before getting started, make sure you have Docker installed. You can pull the TorchServe Docker image from Docker Hub with the following command:
```bash
docker pull pytorch/torchserve:latest-gpu

```
Next, create a virtual environment, and install the required libraries. In your virtual environment, run the following commands:

```bash
# Latest release
pip install torchserve torch-model-archiver torch-workflow-archiver

# install requirement 
pip install -r requirement.txt
```

## Model Archiving

- To serve a PyTorch model with custom handling, you need to organize the following files:
    - `path/to/model-name` [directory]
    - `path/to/model.py`
    - `path/to/checkpoint.pth`
    - `path/to/handler.py`
    - additional files

## Steps 

### 1. Create a Model Store Directory
Let's start by creating a directory to store your models:
```bash
mkdir model_store
```

### 2. Download or allocate a Pre-trained Model
You can download a pre-trained PyTorch model. For example, to get the weights for ResNet-50:

```bash
wget https://download.pytorch.org/models/resnet50-weights.pth
```

### 3. Archive model using model archiver

Use the TorchServe Model Archiver to package your model for serving:
```bash
torch-model-archiver \
    --model-name densenet161 \
    --version 1.0 \
    --model-file <<path/to/model.py>> \
    --serialized-file <<path/to/ckpt.pth>> \
    --export-path <<path/to/model_store>>/ \
    --extra-files <<extra-files>> \
    --handler <<path/to/handler.py>>
```

### 4. Run Docker images as a container : 
Now, you can run the TorchServe Docker container with GPU support:
```bash
docker run --rm -it \
--gpus all \
--name serve \
-p 127.0.0.1:8080:8080 \
-p 127.0.0.1:8081:8081 \
-p 127.0.0.1:8082:8082 \
pytorch/torchserve:latest-gpu
```

### Accessing TorchServe APIs inside container
```bash
curl http://localhost:8080/ping
```

### 5. Access the Container's Bash Prompt
If you need to access the container's shell, use the following command:
```bash
docker exec -it <container_name> /bin/bash

```

### Copy .mar File to the Docker Container
To serve your model, copy the `.mar` file from your local machine to the Docker container:
```bash
docker cp model_store/resnet50.mar serve:/home/model-server/model-store
```
### Register a Model with TorchServe
Register your model with TorchServe using a POST request:
```bash
curl -X POST "localhost:8081/models?model_name=resnet50&url=resnet50.mar&initial_workers=4"
```
### Unregister a Model

To remove a model from TorchServe, use the following command:

```bash
curl -X DELETE "localhost:8081/models/resnet50"
```

