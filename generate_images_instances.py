# This script generates a synthetic dataset of images for object detection,
# mimicking the COCO dataset format. It places various Pokémon sprites (Pikachu,
# Mewtwo, Charizard, Bulbasaur) with random sizes, rotations, and transparencies
# onto different backgrounds. The script ensures a minimum number of sprites per
# image and saves the generated images along with corresponding bounding box
# annotations in a COCO-formatted JSON file.

import os
import cv2
import json
import random
import numpy as np

# --- Configuration ---
SPRITE_FOLDERS = ["pikachu_sprites", "mewtwo_sprites", "charizard_sprites", "bulbasaur_sprites"]
BACKGROUNDS_FOLDER = "Backgrounds_normalized"
OUTPUT_DIR = "scenes"
JSON_PATH = "instances_generated.json"

NUM_IMAGES = 750
IMG_WIDTH, IMG_HEIGHT = 640, 480
MIN_SPRITES_PER_IMAGE = 3 

SIZE_CONFIG = {
    'small': {'range': (40, 80), 'count': (3, 5)},
    'medium': {'range': (80, 150), 'count': (1, 3)},
    'large': {'range': (180, 280), 'count': (1, 1)},
}

# --- Offsets ---
"""Offsets are added make image_id in instances_train start from IMG_OFFSET and id start from ANN_OFFSET"""
IMG_OFFSET = 2000
ANN_OFFSET = 9217 

# ---- Helper: Rotation ----
def rotate_image(img, angle):
    """Rotates an image and expands the BBOX to fit the new dimensions."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - w / 2
    M[1, 2] += (new_h / 2) - h / 2
    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
    return rotated, new_w, new_h

# ---- Helper: Overlay with transparency factor ----
def overlay_image(bg, overlay, x, y, alpha_factor=1.0):
    """Overlays a transparent image onto a background."""
    h, w = overlay.shape[:2]
    # Boundary checks
    if x + w > bg.shape[1]: w = bg.shape[1] - x
    if y + h > bg.shape[0]: h = bg.shape[0] - y
    if x < 0 or y < 0 or w <= 0 or h <= 0: return bg

    overlay = overlay[:h, :w]
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0 * alpha_factor
        for c in range(3):
            bg[y:y + h, x:x + w, c] = (alpha * overlay[:, :, c] +
                                   (1 - alpha) * bg[y:y + h, x:x + w, c])
    return bg

# ---- Helper: Check if bounding boxes overlap ----
def boxes_overlap(a, b, pad=15):
    """Checks for overlap between two boxes with padding."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + pad <= bx or bx + bw + pad <= ax or
                ay + ah + pad <= by or by + bh + pad <= ay)

# ---- Helper: Grid-based placement to find a free spot ----
def find_free_position(placed_boxes, sprite_w, sprite_h, img_w, img_h, grid_size=20):
    """Finds a non-overlapping position for a new sprite using a shuffled grid search."""
    positions = [(x, y) for y in range(0, img_h - sprite_h, grid_size)
                       for x in range(0, img_w - sprite_w, grid_size)]
    random.shuffle(positions)
    for x, y in positions:
        new_box = (x, y, sprite_w, sprite_h)
        if not any(boxes_overlap(new_box, b) for b in placed_boxes):
            return x, y
    return None, None

# ---- Core Function: Scatter sprites on one image ----
def scatter_one(output_path, image_id, ann_start):
    """Generates a single image with sprites placed according to the new logic."""
    bg_files = [f for f in os.listdir(BACKGROUNDS_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not bg_files:
        raise ValueError(f"No background images found in {BACKGROUNDS_FOLDER}")

    bg_file = random.choice(bg_files)
    bg = cv2.imread(os.path.join(BACKGROUNDS_FOLDER, bg_file))
    bg = cv2.resize(bg, (IMG_WIDTH, IMG_HEIGHT))
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)

    annotations = []
    placed_boxes = []
    ann_id = ann_start

    # --- Prepare sprites with size categories ---
    sprites_to_place = []
    for size_cat, config in SIZE_CONFIG.items():
        count = random.randint(config['count'][0], config['count'][1])
        for _ in range(count):
            folder = random.choice(SPRITE_FOLDERS)
            if not os.path.exists(folder): continue
            
            files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
            if not files: continue

            sprite_file = random.choice(files)
            sprite_path = os.path.join(folder, sprite_file)
            sprite = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
            if sprite is None: continue

            # Assign a target size based on the category
            target_size = random.uniform(config['range'][0], config['range'][1])
            sprites_to_place.append({
                "sprite": sprite,
                "folder": folder,
                "category": size_cat,
                "target_size": target_size
            })

    # --- Priority Placement: Sort to place 'large' sprites first ---
    size_order = {'large': 0, 'medium': 1, 'small': 2}
    sprites_to_place.sort(key=lambda s: size_order[s['category']])

    for item in sprites_to_place:
        sprite = item['sprite']
        folder = item['folder']
        target_w = item['target_size']

        # Resize sprite to its target size
        scale = target_w / sprite.shape[1]
        target_h = int(sprite.shape[0] * scale)
        if target_w <= 0 or target_h <= 0: continue
        resized_sprite = cv2.resize(sprite, (int(target_w), target_h), interpolation=cv2.INTER_AREA)

        # Apply random rotation
        angle = random.uniform(0, 360)
        rotated_sprite, rot_w, rot_h = rotate_image(resized_sprite, angle)

        # Find a free position and place the sprite
        sprite_x, sprite_y = find_free_position(placed_boxes, rot_w, rot_h, IMG_WIDTH, IMG_HEIGHT)

        if sprite_x is not None:
            new_box = (sprite_x, sprite_y, rot_w, rot_h)
            alpha_factor = random.uniform(0.75, 1.0)
            bg = overlay_image(bg, rotated_sprite, sprite_x, sprite_y, alpha_factor=alpha_factor)

            folder_to_category_id = {
                "pikachu_sprites": 1,
                "charizard_sprites": 2,
                "bulbasaur_sprites": 3,
                "mewtwo_sprites": 4
            }
            annotations.append({
                "id": ann_id + ANN_OFFSET,
                "image_id": image_id + IMG_OFFSET,
                "category_id": folder_to_category_id[folder],
                "bbox": [sprite_x, sprite_y, rot_w, rot_h],
                "area": rot_w * rot_h,
                "iscrowd": 0
            })
            ann_id += 1
            placed_boxes.append(new_box)
            
    # Save the generated image to the output path
    cv2.imwrite(output_path, bg)
    
    # Return the annotations and the updated annotation counter
    return annotations, ann_id

# ---- Main Function: Generate the full dataset ----
def generate_dataset():
    """Main loop to generate all images and the final JSON file."""
    for folder in SPRITE_FOLDERS + [BACKGROUNDS_FOLDER, OUTPUT_DIR]:
        os.makedirs(folder, exist_ok=True)

    data = {"images": [], "annotations": [], "categories": [
        {"id": 1, "name": "pikachu"},
        {"id": 2, "name": "charizard"},
        {"id": 3, "name": "bulbasaur"},
        {"id": 4, "name": "mewtwo"},
    ]}

    ann_id_counter = 0
    generated_images = 0
    while generated_images < NUM_IMAGES:
        i = generated_images
        file_name = f"img_{i + IMG_OFFSET:05d}.png"
        output_path = os.path.join(OUTPUT_DIR, file_name)
        
        # Keep trying until we have a decent number of sprites in the image.
        attempts = 0
        while True:
            anns, next_ann_id = scatter_one(output_path, i, ann_id_counter)
            
            if len(anns) >= MIN_SPRITES_PER_IMAGE:
                ann_id_counter = next_ann_id # Success: update counter and break
                break

            attempts += 1
            print(f"  - Retrying image {i+1}, only placed {len(anns)} sprites (attempt {attempts}).")
            
            if attempts >= 10: # Failsafe to prevent infinite loops
                print(f"  - Warning: Failed to place enough sprites for image {i} after 10 attempts. Skipping.")
                anns = [] # Ensure this image is skipped
                break
        
        # If generation failed completely after retries, don't increment the image count
        if not anns:
            continue

        data["images"].append({
            "id": i + IMG_OFFSET,
            "file_name": file_name,
            "width": IMG_WIDTH,
            "height": IMG_HEIGHT
        })
        data["annotations"].extend(anns)
        generated_images += 1 # Only increment on a successful generation

    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)
        
    print(f"\nSuccessfully generated {len(data['images'])} images and saved JSON to {JSON_PATH}")

# --- Run the script ---
if __name__ == "__main__":
    generate_dataset()

