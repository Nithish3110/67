# This script enhances images by adding cinematic visual effects such as bokeh,
# lens flares, and glitter particles. It randomly generates effect positions,
# sizes, colors, and intensities to produce natural-looking overlays. The script
# processes all supported images in an input folder and saves the enhanced
# outputs to the specified output folder, while robustly handling errors or
# mismatched effect sizes during overlay.

import cv2
import numpy as np
import os
import random
from pathlib import Path

# --- BOKEH EFFECT FUNCTIONS ---

def create_bokeh_circle(radius, color=(255, 200, 100), intensity=0.8):
    size = radius * 2
    center = radius
    bokeh = np.zeros((size, size, 4), dtype=np.float32)
    y, x = np.ogrid[:size, :size]
    distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    distance = distance / radius
    alpha = np.exp(-distance ** 2 * 3)
    alpha = np.clip(alpha, 0, 1)
    bokeh[:, :, 0] = color[0] * alpha * intensity
    bokeh[:, :, 1] = color[1] * alpha * intensity
    bokeh[:, :, 2] = color[2] * alpha * intensity
    bokeh[:, :, 3] = alpha * 255
    return bokeh.astype(np.uint8)

def generate_bokeh_positions(image_shape, num_bokeh=(30, 55)):
    height, width = image_shape[:2]
    positions = []
    num_circles = random.randint(num_bokeh[0], num_bokeh[1])
    for _ in range(num_circles):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        edge_factor = min(x, width - x, y, height - y) / min(width, height) * 4
        base_radius = random.randint(30, 80)
        radius = int(base_radius * (1 + edge_factor * 0.5))
        radius = min(radius, 200)
        positions.append((x, y, radius))
    return positions

# --- LENS FLARE AND GLITTER FUNCTIONS ---

def create_lens_flare(size, angle, color=(255, 255, 220), intensity=0.5):
    flare_canvas = np.zeros((size, size, 4), dtype=np.float32)
    center = size // 2
    streak_height = max(1, int(size / 100))
    y, x = np.ogrid[-center:center, -center:center]
    falloff = np.exp(-(x ** 2) / (size * 20) - (y ** 2) / (streak_height * 2))
    flare_canvas[:, :, 0] = color[0] * falloff * intensity
    flare_canvas[:, :, 1] = color[1] * falloff * intensity
    flare_canvas[:, :, 2] = color[2] * falloff * intensity
    flare_canvas[:, :, 3] = falloff * 255
    M = cv2.getRotationMatrix2D((center, center), angle, 1)
    rotated_flare = cv2.warpAffine(flare_canvas, M, (size, size))
    return rotated_flare.astype(np.uint8)

def create_glitter(size=10, color=(255, 255, 240), intensity=0.9):
    glitter = np.zeros((size, size, 4), dtype=np.float32)
    center = size // 2
    cv2.line(glitter, (0, center), (size - 1, center), (*color, 255), 1)
    cv2.line(glitter, (center, 0), (center, size - 1), (*color, 255), 1)
    cv2.line(glitter, (0, 0), (size - 1, size - 1), (*color, 255), 1)
    cv2.line(glitter, (size - 1, 0), (0, size - 1), (*color, 255), 1)
    glitter = cv2.GaussianBlur(glitter, (5, 5), 0)
    glitter[:, :, :3] *= intensity
    glitter[:, :, 3] *= intensity
    return np.clip(glitter, 0, 255).astype(np.uint8)

# --- ROBUST OVERLAY FUNCTION (SKIP ON ERROR OR SHAPE MISMATCH) ---

def overlay_effect(image, effect, position, blend_strength=0.7):
    try:
        img_h, img_w = image.shape[:2]
        effect_h, effect_w = effect.shape[:2]
        x_center, y_center = position
        x1 = x_center - effect_w // 2
        x2 = x1 + effect_w
        y1 = y_center - effect_h // 2
        y2 = y1 + effect_h

        img_y1_slice = max(0, y1)
        img_y2_slice = min(img_h, y2)
        img_x1_slice = max(0, x1)
        img_x2_slice = min(img_w, x2)

        if img_y2_slice <= img_y1_slice or img_x2_slice <= img_x1_slice:
            return image

        eff_y1_slice = max(0, -y1)
        eff_x1_slice = max(0, -x1)

        overlay_h = img_y2_slice - img_y1_slice
        overlay_w = img_x2_slice - img_x1_slice

        eff_y2_slice = eff_y1_slice + overlay_h
        eff_x2_slice = eff_x1_slice + overlay_w

        roi = image[img_y1_slice:img_y2_slice, img_x1_slice:img_x2_slice]
        effect_slice = effect[eff_y1_slice:eff_y2_slice, eff_x1_slice:eff_x2_slice]

        # Shape check -- if not matching, skip overlay and return image
        if roi.shape[:2] != effect_slice.shape[:2]:
            return image

        roi_float = roi.astype(np.float32)
        effect_rgb = effect_slice[:, :, :3].astype(np.float32)
        alpha = (effect_slice[:, :, 3].astype(np.float32) / 255.0) * blend_strength

        for c in range(3):
            roi_float[:, :, c] += effect_rgb[:, :, c] * alpha

        image[img_y1_slice:img_y2_slice, img_x1_slice:img_x2_slice] = np.clip(roi_float, 0, 255).astype(np.uint8)
        return image

    except Exception:
        # Silently skip overlay step in case of error
        return image

# --- MAIN PROCESSING FUNCTIONS ---

def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    height, width = image.shape[:2]

    # 1. Apply Bokeh Effect
    positions = generate_bokeh_positions(image.shape)
    bokeh_colors = [
        (255, 182, 193), (255, 218, 185), (238, 130, 238),
        (0, 0, 139), (0, 255, 255),
    ]
    for x, y, radius in positions:
        color = random.choice(bokeh_colors)
        intensity = random.uniform(0.3, 0.8)
        bokeh_circle = create_bokeh_circle(radius, color, intensity)
        image = overlay_effect(image, bokeh_circle, (x, y), blend_strength=0.7)

    # 2. Chance for lens flare and glitter
    if random.random() < 0.2:
        flare_size = int(max(width, height) * random.uniform(0.8, 1.2))
        flare_angle = random.uniform(0, 360)
        flare_intensity = random.uniform(0.2, 0.4)
        flare_x = random.randint(0, width)
        flare_y = random.randint(0, height)
        lens_flare = create_lens_flare(flare_size, flare_angle, intensity=flare_intensity)
        image = overlay_effect(image, lens_flare, (flare_x, flare_y), blend_strength=0.6)
        num_glitter = random.randint(70, 150)
        for _ in range(num_glitter):
            glitter_x = random.randint(0, width - 1)
            glitter_y = random.randint(0, height - 1)
            glitter_particle = create_glitter(size=random.randint(8, 14))
            image = overlay_effect(image, glitter_particle, (glitter_x, glitter_y), blend_strength=0.5)

    cv2.imwrite(output_path, image)
    print(f"Processed: {os.path.basename(image_path)} -> {os.path.basename(output_path)}")

def process_folder(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [fn for fn in os.listdir(input_folder) if Path(fn).suffix.lower() in supported_extensions]
    if not image_files:
        print(f"No supported image files found in {input_folder}")
        return
    print(f"Found {len(image_files)} images to process...")
    try:
        image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    except ValueError:
        image_files.sort()

    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        try:
            process_image(input_path, output_path)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    print(f"\nProcessing complete! Output saved to: {output_folder}")

# --- SCRIPT EXECUTION ---

if __name__ == "__main__":
    INPUT_FOLDER = "scenes"
    OUTPUT_FOLDER = "scenes"
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' not found!")
    else:
        process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
