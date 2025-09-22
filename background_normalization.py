import cv2
import os

def convert_and_rename(input_folder, output_folder, size=(640, 480)):
    # Make output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    # Collect only .jpg files
    jpg_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")]

    for idx, file in enumerate(jpg_files, start=1):
        img_path = os.path.join(input_folder, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping {file}, could not load.")
            continue

        # Resize
        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        # Save as background_X.png
        output_path = os.path.join(output_folder, f"background_{idx}.png")
        cv2.imwrite(output_path, resized)

        print(f"Converted {file} -> {output_path}")


# Example usage:
convert_and_rename("Backgrounds", "Backgrounds_normalized")
