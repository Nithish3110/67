# CELL: YOLO + SAM integration
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

# --- User-editable paths / params ---
YOLO_WEIGHTS = "yolov8x.pt"                 # or path to your trained YOLO model
SAM_CHECKPOINT = "/kaggle/input/sam-checkpoints/sam_vit_h_4b8939.pth"  # path to SAM checkpoint
SAM_MODEL_TYPE = "vit_h"                    # "vit_h", "vit_l", "vit_b" depending on checkpoint
IMAGES_PATH = IMAGES_PATH                   # uses variable from your earlier cell
OUTPUT_DIR = "/kaggle/working/yolo_sam_outputs"
CONF_THRESHOLD = 0.35                       # YOLO confidence threshold to keep boxes
USE_AUTOMATIC_MASK_GENERATOR = False        # If True uses SAM AutomaticMaskGenerator (no prompts)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load models ---
print("Loading YOLO model...")
yolo_model = YOLO(YOLO_WEIGHTS)

print("Loading SAM model...")
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# optional automatic mask generator (no prompt)
sam_automatic = SamAutomaticMaskGenerator(sam) if USE_AUTOMATIC_MASK_GENERATOR else None

def read_bgr(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return img

def xyxy_to_box_prompt(x1, y1, x2, y2, img_h, img_w):
    # SAM expects box in [x1, y1, x2, y2] (absolute pixel coords)
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def visualize_and_save(img_bgr, boxes, masks, classes=None, output_path=None, overlay_alpha=0.5):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_rgb)
    # draw boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        w, h = (x2 - x1), (y2 - y1)
        rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
        if classes:
            ax.text(x1, y1 - 8, str(classes[i]), color='yellow', fontsize=12, weight='bold')
    # overlay masks
    for i, mask in enumerate(masks):
        colored_mask = np.zeros_like(img_rgb, dtype=np.uint8)
        color = np.random.randint(0, 255, 3)
        colored_mask[mask == 1] = color
        ax.imshow(np.dstack([colored_mask, (mask * 255).astype(np.uint8)])[:,:, :3], alpha=overlay_alpha)
    ax.axis('off')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)

def yolo_then_sam_on_image(image_path, save=True, visual=True):
    """Run YOLO to detect objects, then run SAM using box prompts for each detection."""
    img = read_bgr(image_path)
    h, w = img.shape[:2]
    # YOLO inference
    results = yolo_model.predict(source=str(image_path), imgsz=640, conf=CONF_THRESHOLD, verbose=False)
    # collect boxes and classes
    boxes = []
    cls_ids = []
    masks = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            boxes.append([x1, y1, x2, y2])
            cls_ids.append(cls)
    if len(boxes) == 0 and sam_automatic is None:
        print("No detections from YOLO (or detection threshold too high).")
    # If using SAM automatic generator, use it on the whole image
    if sam_automatic is not None:
        print("Using SAM AutomaticMaskGenerator on entire image...")
        sam_masks = sam_automatic.generate(img[:,:,::-1])  # expects RGB
        # each sam_mask is a dict with 'segmentation' (boolean mask)
        for m in sam_masks:
            masks.append(m['segmentation'].astype(np.uint8))
    else:
        # Use SAM predictor with box prompts per detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(img_rgb)
        for box in boxes:
            box_np = xyxy_to_box_prompt(*box, img_h=h, img_w=w)
            # predict with box prompt; returns masks (n, h, w), scores...
            masks_result, _, _ = sam_predictor.predict(
                box=box_np.reshape(1, 4),
                multimask_output=False
            )
            # masks_result is (1, H, W)
            if masks_result is not None and len(masks_result) > 0:
                masks.append((masks_result[0] > 0).astype(np.uint8))

    out_path = None
    if save or visual:
        fname = Path(image_path).stem
        out_path = os.path.join(OUTPUT_DIR, f"{fname}_yolo_sam_vis.png")
        if visual:
            visualize_and_save(img, boxes, masks, classes=cls_ids, output_path=out_path)
    return {"image": str(image_path), "boxes": boxes, "classes": cls_ids, "masks": masks, "vis_path": out_path}

# Usage: run on first image in IMAGES_PATH
example_images = [p for p in Path(IMAGES_PATH).glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
if len(example_images) > 0:
    result = yolo_then_sam_on_image(example_images[0], save=True, visual=True)
    print("Result keys:", result.keys())
else:
    print("No images found in", IMAGES_PATH)

# CELL: Run SAM alone (automatic mask generation or box prompt)
import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from pathlib import Path

# --- User-editable ---
SAM_CHECKPOINT = "/kaggle/input/sam-checkpoints/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"     # choose matching variant of checkpoint
IMAGE_PATH = "/kaggle/input/actualtest1/some_image.jpg"  # replace with real image path
USE_AUTOMATIC = True         # True: AutomaticMaskGenerator (no prompts). False: box prompt example.
BOX_PROMPT = [50, 30, 300, 240]  # example [x1,y1,x2,y2] if USE_AUTOMATIC=False
OUTPUT_DIR = "/kaggle/working/sam_only_outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load SAM ---
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)

# Automatic mask generator
if USE_AUTOMATIC:
    print("Using SAM AutomaticMaskGenerator...")
    mask_generator = SamAutomaticMaskGenerator(sam)
    img_bgr = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(img_rgb)  # list of dicts {segmentation, area, bbox, ...}
    print(f"Generated {len(masks)} masks.")
    # visualize first N masks
    def show_masks_on_image(img_bgr, masks, max_masks=10, alpha=0.5):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img_rgb)
        for i, m in enumerate(masks[:max_masks]):
            seg = m['segmentation'].astype(bool)
            colored = np.zeros_like(img_rgb, dtype=np.uint8)
            color = np.random.randint(0, 255, 3)
            colored[seg] = color
            ax.imshow(colored, alpha=alpha)
            x, y, w, h = m['bbox']
            rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='white', facecolor='none')
            ax.add_patch(rect)
        ax.axis('off')
        plt.show()
    show_masks_on_image(img_bgr, masks, max_masks=8)
else:
    # Predictor with a box prompt
    predictor = SamPredictor(sam)
    img_bgr = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    box = np.array(BOX_PROMPT, dtype=np.float32).reshape(1, 4)
    masks, scores, logits = predictor.predict(box=box, multimask_output=False)
    mask = masks[0]  # boolean mask
    print("Mask shape:", mask.shape, "Score:", scores)
    # visualize
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img_rgb)
    colored = np.zeros_like(img_rgb, dtype=np.uint8)
    colored[mask] = (0, 255, 0)
    ax.imshow(colored, alpha=0.5)
    x1, y1, x2, y2 = BOX_PROMPT
    ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none', linewidth=2))
    ax.axis('off')
    plt.show()
