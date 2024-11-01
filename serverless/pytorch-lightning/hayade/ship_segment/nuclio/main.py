import json
import base64
from PIL import Image
import io
import numpy as np
import cv2
import torch

from components.unet_module import UNetLitModule
from components.unet import UNet
from components.loss_binary import LossBinary

from utils import preprocess, to_cvat_mask
from skimage.measure import find_contours, approximate_polygon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_context(context):
    context.logger.info(f"Using {device}")
    context.logger.info("Init context...  0%")

    model = UNetLitModule.load_from_checkpoint(
        "unet.ckpt",
        net=UNet(),
        criterion=LossBinary(),
    )

    context.user_data.model = model
    context.user_data.model = context.user_data.model.to(device)

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run UNet model")
    data = event.body

    # read image
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf).convert("RGB")

    # predict the segment mask
    mask = context.user_data.model(preprocess(image).to(device)).detach().squeeze()
    mask = torch.sigmoid(mask)

    confidence = mask[mask >= 0.5].mean().item()

    # postprocess
    mask = (
        (mask >= 0.5).cpu().numpy().astype(np.uint8)
    )
    mask = cv2.resize(mask, image.size)

    contours = find_contours(mask)
    results = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        contour = approximate_polygon(contour, tolerance=2.5)

        Xmin = int(np.min(contour[:,0]))
        Xmax = int(np.max(contour[:,0]))
        Ymin = int(np.min(contour[:,1]))
        Ymax = int(np.max(contour[:,1]))
        cvat_mask = to_cvat_mask((Xmin, Ymin, Xmax, Ymax), mask)

        results.append({
            "confidence": f"{confidence:.2f}",
            "label": "ship",
            'points': np.array(contour).flatten().tolist(),
            "mask": cvat_mask,
            "type": "mask",

        })

        # print("confidence:", f"{confidence:.2f}")
        # print("points:", np.array(contour).flatten().tolist())
        # print("mask:", cvat_mask)

    return context.Response(body=json.dumps(results),
        headers={},
        content_type='application/json',
        status_code=200
    )

