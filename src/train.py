# Import libraries
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.models as models
from engine import train_one_epoch, evaluate
import utils
from data import CellDataset
from ECAResnet50 import eca_resnet50, TimmToVisionFPN
from SEResnet50 import se_resnet50
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import MaskRCNN
import os
import cv2
import numpy as np
import random
import datetime
import json



def get_transform(train):
    """
    Define a list of data augmentation and transformation operations
    Args:
        train (bool): A boolean flag indicating whether the transformations are for training (True) or evaluation (False).
    Returns:
        torchvision.transforms.Compose: A composition of data augmentation and preprocessing transformations.
    """

    transforms = []

    # If 'train' is True, add random horizontal flip with a 50% probability for data augmentation.
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    # Convert the image data to torch.float and scale it.
    transforms.append(T.ToDtype(torch.float, scale=True))

    # Convert the image data to a pure tensor.
    transforms.append(T.ToPureTensor())

    # Combine all the transformations into a single Compose transformation.
    return T.Compose(transforms)


def get_model_instance_segmentation(num_classes, name):
    """
    Create an instance segmentation model with a specified backbone architecture.
    Args:
        num_classes (int): The number of classes for the instance segmentation model.
        name (str): Name of the backbone architecture, which can be one of:
            - 'maskrcnn_resnet50_fpn': ResNet-50 backbone with FPN.
            - 'maskrcnn_eca-resnet50_fpn': ECA-ResNet-50 backbone with FPN.
            - 'maskrcnn_se-resnet50_fpn': SE-ResNet-50 backbone with FPN.

    Returns:
        torchvision.models.detection.mask_rcnn.MaskRCNN: The instance segmentation model with the specified backbone.

    Raises:
        ValueError: If the provided 'name' is not one of the valid options.
    """
    if name == 'maskrcnn_resnet50_fpn':
        # Create a Mask R-CNN model with ResNet-50 backbone and FPN.
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # replacing the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256

        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes,
        )

        return model

    elif name == 'maskrcnn_eca-resnet50_fpn':
        # Create a Mask R-CNN model with ECA-ResNet-50 backbone and FPN.
        backbone = TimmToVisionFPN(eca_resnet50())

        # Define the RoI pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],  # Adjust based on your backbone
            output_size=7,
            sampling_ratio=2
        )

        model = MaskRCNN(
            backbone,
            num_classes=2,  # Adjust the number of classes as needed
            box_roi_pool=roi_pooler
        )

        return model

    elif name == 'maskrcnn_se-resnet50_fpn':
        # Create a Mask R-CNN model with SE-ResNet-50 backbone and FPN.
        backbone = TimmToVisionFPN(se_resnet50())

        # Define the RoI pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],  # Adjust based on your backbone
            output_size=7,
            sampling_ratio=2
        )

        model = MaskRCNN(
            backbone,
            num_classes=2,  # Adjust the number of classes as needed
            box_roi_pool=roi_pooler
        )

        return model

    else:
        raise ValueError("the name value is not valid it should be ['maskrcnn_resnet50_fpn',"
                         "'maskrcnn_eca-resnet50_fpn', 'maskrcnn_se-resnet50_fpn']")


def train(model_name, epochs, num_classes=2, data_dir='data',
          test_json_file='data/test.json', log_dir="./logs",
          model_output_dir='models',
          results_dir='results'):
    writer = SummaryWriter(log_dir)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # define validation and train dataset
    dataset = CellDataset(data_dir, get_transform(train=True))
    dataset_val = CellDataset(data_dir, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes, name=model_name)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # train model
    for epoch in range(epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the val dataset

        # Validation loop
        evaluate(model, data_loader_val, device=device)

    print("Tha training is done!")
    writer.close()

    # Creating a directory for saving models
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    # Creating a directory for metrics
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Get the current date and time
    current_time = datetime.datetime.now()

    # Format the date and time as a string (e.g., "2023-10-21_14-30-15")
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Save the model
    torch.save(model.state_dict(), os.path.join(model_output_dir, f'{model_name}_{timestamp}.pth'))

    # Create a loop to visualize prediction on unseen data (test dataset)
    num_samples_to_display = 6
    images_dir = os.path.join(data_dir, 'images', 'test')
    data = CellDataset(data_dir, get_transform(train=False), stage='test')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    with open(test_json_file, 'r') as test_file:
        test_data = json.load(test_file)

    for i in range(num_samples_to_display):
        # Randomly select an image
        idx = random.randint(0, len(test_data['images']) - 1)
        image_info = test_data['images'][idx]
        image_path = os.path.join(images_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = data.create_mask(image_info, test_data['annotations'])

        # Resize the image and mask
        image = torch.from_numpy(image).permute(2, 0, 1)
        eval_transform = get_transform(train=False)
        mask = torch.from_numpy(mask)
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        real_masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
        real_masks = real_masks.bool()

        model.eval()
        with torch.no_grad():
            x = eval_transform(image)
            x = x[:3, ...].to(device)
            predictions = model([x, ])
            pred = predictions[0]

        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        image = image[:3, ...]

        masks = (pred["masks"] > 0.7).squeeze(1)
        output_image = draw_segmentation_masks(image, masks, alpha=0.5, colors="blue")
        output_image = draw_segmentation_masks(output_image, real_masks, alpha=0.7, colors="green")

        # Plot images and masks in the current subplot
        row = i // 3
        col = i % 3
        axes[row, col].imshow(output_image.permute(1, 2, 0))
        axes[row, col].set_title(f"Sample {i + 1}")

    # Remove x and y ticks from subplots
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    # Show the subplots
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name}_pred_{timestamp}.png'), dpi=300)
    plt.show()


models_name = ['maskrcnn_resnet50_fpn', 'maskrcnn_eca-resnet50_fpn', 'maskrcnn_se-resnet50_fpn']
for model_name in models_name:
    train(model_name, epochs=10)