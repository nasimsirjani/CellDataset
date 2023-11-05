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







