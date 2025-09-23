# CELL: YOLO + GrabCut integration
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

# --- User-editable paths / params ---
YOLO_WEIGHTS = "yolov8x.pt"                 # or path to your trained YOLO model
IMAGES_PATH = IMAGES_PATH                   # reusing variable from earlier
OUTPUT_DIR = "/kaggle/working/yolo_grabcut_outputs"
CONF_THRESHOLD = 0.35
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load YOLO model ---
print("Loading YOLO model...")
yolo_model = YOLO(YOLO_WEIGHTS)

def read_bgr(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return img

def apply_grabcut(img_bgr, box, iter_count=5):
    """
    Apply GrabCut given a bounding box.
    box: [x1, y1, x2, y2]
    Returns binary mask (0/1).
    """
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Convert to (x, y, w, h) for GrabCut rect input
    x1, y1, x2, y2 = map(int, box)
    rect = (x1, y1, x2 - x1, y2 - y1)

    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)

    # Convert mask to binary foreground mask
    output_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    return output_mask

def visualize_and_save(img_bgr, boxes, masks, classes=None, output_path=None, overlay_alpha=0.5):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_rgb)

    # Draw boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
        if classes:
            ax.text(x1, y1 - 8, str(classes[i]),
                    color='yellow', fontsize=12, weight='bold')

    # Overlay masks
    for i, mask in enumerate(masks):
        colored_mask = np.zeros_like(img_rgb, dtype=np.uint8)
        color = np.random.randint(0, 255, 3)
        colored_mask[mask == 1] = color
        ax.imshow(colored_mask, alpha=overlay_alpha)

    ax.axis('off')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)

def yolo_then_grabcut_on_image(image_path, save=True, visual=True):
    """Run YOLO detection, then GrabCut segmentation on each box."""
    img = read_bgr(image_path)

    results = yolo_model.predict(source=str(image_path), imgsz=640,
                                 conf=CONF_THRESHOLD, verbose=False)

    boxes, cls_ids, masks = [], [], []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            boxes.append([x1, y1, x2, y2])
            cls_ids.append(cls)

            # Apply GrabCut on this detection
            mask = apply_grabcut(img, [x1, y1, x2, y2])
            masks.append(mask)

    out_path = None
    if save or visual:
        fname = Path(image_path).stem
        out_path = os.path.join(OUTPUT_DIR, f"{fname}_yolo_grabcut_vis.png")
        if visual:
            visualize_and_save(img, boxes, masks, classes=cls_ids, output_path=out_path)

    return {"image": str(image_path), "boxes": boxes,
            "classes": cls_ids, "masks": masks, "vis_path": out_path}

# Example usage: run on first image in IMAGES_PATH
example_images = [p for p in Path(IMAGES_PATH).glob("*")
                  if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
if len(example_images) > 0:
    result = yolo_then_grabcut_on_image(example_images[0], save=True, visual=True)
    print("Result keys:", result.keys())
else:
    print("No images found in", IMAGES_PATH)
