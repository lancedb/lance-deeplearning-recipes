
import lancedb
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
import torch
import pickle
from typing import Optional
from pycocotools import mask as coco_mask
import matplotlib.patches as patches


def coordinates_to_mask(pickled_segmentation: bytes, height: int, width: int) -> Optional[np.ndarray]:
    """
    Converts pickled COCO segmentation data into a binary mask.
    """
    if not pickled_segmentation:
        return None
    
    segmentation = pickle.loads(pickled_segmentation)
    
    try:
        if isinstance(segmentation, list):
            rles = coco_mask.frPyObjects(segmentation, height, width)
            rle = coco_mask.merge(rles)
            mask = coco_mask.decode(rle)
        elif isinstance(segmentation, dict):
            mask = coco_mask.decode(segmentation)
        else:
            return None
        return mask.astype(np.uint8)
    except Exception as e:
        logger.warning(f"Could not decode segmentation mask. Error: '{e}'.")
        return None

    
def display_result(row):
    """
    ### UPDATED ###
    Displays an image with its captions, instance masks, AND bounding boxes.
    """
    image_bytes = row['image_bytes']
    if not image_bytes:
        print("No image data for this result.")
        return

    image = Image.open(io.BytesIO(image_bytes))
    plt.figure(figsize=(12, 10))
    ax = plt.gca()  # Get current axes
    ax.imshow(image)
    ax.axis('off')

    # --- Overlay instance masks ---
    if 'instance_masks' in row and row['instance_masks'] is not None:
        for mask_obj in row['instance_masks']:
            if mask_obj and 'segmentation' in mask_obj:
                mask_data = mask_obj['segmentation']
                mask = coordinates_to_mask(mask_data, row['height'], row['width'])
                if mask is not None:
                    # Create a colored overlay
                    colored_mask = np.zeros((*mask.shape, 4))
                    # Use a random color for each mask for better distinction
                    color = np.random.random(3)
                    colored_mask[mask == 1] = np.concatenate([color, [0.5]]) # Color with 50% opacity
                    ax.imshow(colored_mask)

    # --- Overlay bounding boxes ---
    if 'bounding_boxes' in row and row['bounding_boxes'] is not None:
        for bbox in row['bounding_boxes']:
            # COCO format is [x_min, y_min, width, height]
            x, y, w, h = bbox
            # Create a Rectangle patch
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor='lime',  # A bright color to stand out
                facecolor='none'     # No fill
            )
            # Add the patch to the Axes
            ax.add_patch(rect)
    
    # Print captions below the image
    captions_text = "\n".join(f"- {c.strip()}" for c in row['captions'])
    plt.title(f"Image ID: {row['image_id']}\nCaptions:\n{captions_text}", loc='left', wrap=True, fontsize=10)
    
    plt.show()