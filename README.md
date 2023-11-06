# LIVECell dataset segmentation

Object segmentation in LIVECell dataset by using MaskRCNN with three different backbones:
  - maskrcnn_resnet50_fpn
  - maskrcnn_eca-resnet50_fpn
  - askrcnn_se-resnet50_fpn
    
## Introduction

This project aims to segment cells in an image by using MaskRCNN. It is designed to demonstrate the effect of different backbones in MaskRCNN structure.

## Features
- Image object segmentation (LIVECell dataset)
- Data visualization
- Model training and evaluation
- Docker containerization for easy deployment and reproducibility
- Automated CI/CD pipeline using GitHub Actions
- Heroku deployment for online prediction

## Directories
- data: this directory includes our dataset 
- logs: this directory includes our training logs
- models: this directory includes our models' checkpoints
- notebook: this directory includes two .ipynb that one of them is about data visualization phase and the another one is about train-test phase
- results: this directory includes of 2D images that shows the prediction of models for six random samples
- src: this directory includes all necessary codes that we need
- src/templates: this directory includes .html files for the app

## Getting Started

### Prerequisites

Before you begin, make sure you have Docker installed on your system. You can install Docker by following the instructions on the [official Docker website](https://docs.docker.com/get-docker/).

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/nasimsirjani/CellDataset.git
   cd CellDataset
2. run the command docker-compose -f .\docker-compose-train.yaml up --build -d
3. run the command docker-compose -f .\docker-compose-app.yaml up --build by running this command you can connect to the http://localhost:8080/process_image for online prediction
4. run by kubernetes:
   - kubectl apply -f training-deployment.yaml
   - kubectl apply -f web-deployment.yaml
5. by triggering the .github/workflows/CICD training and deploying will happen automatically but because I did not have pgu cloud on which run it, it is time-consuming
6. by triggering the .github/workflows/CICD



