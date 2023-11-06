# Import libraries
import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torchvision.models as models
from engine import train_one_epoch, evaluate
import utils
from data import CellDataset
from ECAResnet50 import eca_resnet50
from Backend import get_model_instance_segmentation, get_transform
from SEResnet50 import se_resnet50
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
import numpy as np
import random
import datetime
import json


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

    # split the dataset in train and val set
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