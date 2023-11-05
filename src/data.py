import os
import torch
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import json
import cv2
import numpy as np


class CellDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, stage='train'):
        self.root = root
        self.transforms = transforms
        self.images_dir = os.path.join(root, 'images', stage)
        self.desired_size = (224, 224)

        # Load JSON files
        json_file = os.path.join(root, f'{stage}.json')
        with open(json_file, 'r') as file_:
            self.data = json.load(file_)

    def create_mask(self, image_info, annotations):
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        object_id = 1
        for annotation in annotations:
            try:
                if annotation['image_id'] == image_info['id'] and annotation['iscrowd'] == 0:
                    # Extract segmentation polygon
                    for polygon in annotation['segmentation']:
                        if len(polygon) % 2 != 0:
                            continue
                        x_coordinates = polygon[::2]
                        y_coordinates = polygon[1::2]

                        points = np.array([[int(x), int(y)] for x, y in zip(x_coordinates, y_coordinates)])

                        # Reshape the points to match the required format for cv2.fillPoly
                        points = points.reshape((-1, 1, 2))

                        # Draw the polygon on the image
                        cv2.fillPoly(mask, [points], (object_id))  # (0, 255, 0) is the color in BGR format
                        object_id += 1
            except:
                continue

        return np.array(mask)

    def __getitem__(self, idx):
        # load images and masks
        image_info = self.data['images'][idx]
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = self.create_mask(image_info, self.data['annotations'])

        # Resize the image and mask
        img = cv2.resize(img, self.desired_size)
        mask = cv2.resize(mask, self.desired_size,
                          interpolation=cv2.INTER_NEAREST)

        # Convert image and mask to PyTorch tensors
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask)

        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        boxes = masks_to_boxes(masks)

        # Filter out invalid bounding boxes with negative or zero width/height
        valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_boxes]

        # Filter out corresponding masks and labels
        masks = masks[valid_boxes]

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.data['images'])