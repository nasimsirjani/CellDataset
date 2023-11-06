from collections import OrderedDict
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch import nn
from torchvision.transforms import v2 as T
import torchvision
from torchvision.models.detection import MaskRCNN
import torch


class TimmToVisionFPN(nn.Module):
    def __init__(self, backbone):
        super(TimmToVisionFPN, self).__init__()
        self.backbone = backbone
        self.out_channels = 256
        self.in_channels_list = [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, x):
        layer1 = self.backbone.conv1(x)
        layer1 = self.backbone.bn1(layer1)
        layer1 = self.backbone.relu(layer1)
        layer1 = self.backbone.maxpool(layer1)
        layer2 = self.backbone.layer1(layer1)
        layer3 = self.backbone.layer2(layer2)
        layer4 = self.backbone.layer3(layer3)
        layer5 = self.backbone.layer4(layer4)
        x = [layer1, layer2, layer3, layer4, layer5]
        out = OrderedDict()
        for i in range(len(x)-1):
            out[str(i)] = x[i+1]
        out = self.fpn(out)
        return out



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