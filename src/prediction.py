import torch
import torchvision
from ECAResnet50 import eca_resnet50
from SEResnet50 import se_resnet50
from torchvision.transforms import v2 as T
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import json
import random
import os
import cv2


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


# Define the path to the checkpoint file
checkpoint_file = '/content/models/maskrcnn_resnet50_fpn_2023-11-05_15-38-27.pth'

# Create an instance of the CustomMaskRCNN model
model = get_model_instance_segmentation(name='maskrcnn_resnet50_fpn', num_classes=2)  # Adjust name and num_classes

# Load the state dictionary with the mapped keys
model.load_state_dict(torch.load(checkpoint_file))
model.eval()

# You don't need to set the model to evaluation mode if you're not providing targets

test_json_file = '/content/data/test.json'
with open(test_json_file, 'r') as test_file:
    test_data = json.load(test_file)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
images_dir = '/content/data/images/test'
idx = random.randint(0, len(test_data['images'])-1)
image_info = test_data['images'][idx]
image_path = os.path.join(images_dir, image_info['file_name'])
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = torch.from_numpy(image).permute(2, 0, 1)
eval_transform = get_transform(train=False)


with torch.no_grad():
    x = eval_transform(image)
    x = x[:3, ...]
    predictions = model([x, ])
    pred = predictions[0]

image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
image = image[:3, ...]

masks = (pred["masks"] > 0.7).squeeze(1)
output_image = draw_segmentation_masks(image, masks, alpha=0.5, colors="blue")

plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))