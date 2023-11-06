from flask import Flask, request, jsonify, render_template
import torch
import cv2
import os
import random
import json
import numpy as np
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from ECAResnet50 import eca_resnet50
from SEResnet50 import se_resnet50
from Backend import get_model_instance_segmentation, get_transform
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


# Create app
app = Flask(__name__)


# Define the path to the checkpoint file
checkpoint_file = 'models/maskrcnn_resnet50_fpn_2023-11-05_17-13-41.pth'

# Load model
model = get_model_instance_segmentation(name='maskrcnn_resnet50_fpn', num_classes=2)  # Adjust name and num_classes
# Ensure the model is loaded on the CPU if no GPU is available
if not torch.cuda.is_available():
    model.load_state_dict(torch.load(checkpoint_file, map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(checkpoint_file))
model.eval()


def process_image(image):
    try:
        # Convert the input image to a PyTorch tensor
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Apply the evaluation data transformation
        eval_transform = get_transform(train=False)

        with torch.no_grad():
            x = eval_transform(image)
            x = x[:3, ...]

            # Make predictions using the model
            predictions = model([x, ])
            pred = predictions[0]

        # Normalize and prepare the input image
        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        image = image[:3, ...]

        # Generate masks based on model predictions
        masks = (pred["masks"] > 0.7).squeeze(1)

        # Overlay the masks on the input image for visualization
        output_image = draw_segmentation_masks(image, masks, alpha=0.5, colors="blue")

        return output_image

    except Exception as e:
        return None


@app.route("/")
def index():
    return render_template("app_template.html")


@app.route("/process_image", methods=["POST"])
def process_image_route():
    try:
        # Extract image data from the POST request
        image_data = request.files["image"]
        image = cv2.imdecode(np.fromstring(image_data.read(), np.uint8), cv2.IMREAD_COLOR)

        # Prepare the output
        result_image = process_image(image)

        if result_image is not None:
            result_image = result_image.permute(1, 2, 0).numpy().astype(np.uint8)
            # Convert the result to a base64-encoded image
            _, buffer = cv2.imencode(".png", result_image)
            result_image_base64 = base64.b64encode(buffer).decode("utf-8")

            return render_template("result.html", result_image_base64=result_image_base64)
        else:
            return jsonify({"error": "Image processing failed"})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
