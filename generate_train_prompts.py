import json

# Load the instances_generated.json file
with open("instances_generated.json", "r") as f:
    data = json.load(f)

# Create id to name mapping from 'categories'
category_map = {cat['id']: cat['name'] for cat in data['categories']}

# Prepare annotation lookup: image_id to list of annotations
annotation_map = {}
for ann in data['annotations']:
    annotation_map.setdefault(ann['image_id'], []).append(ann)

prompts = []
for img in data['images']:
    image_id_str = img['file_name']
    img_id = img['id']
    
    # Get the first annotation for this image (or skip if none)
    anns = annotation_map.get(img_id, [])
    if not anns:
        continue
    
    # Use the class of the first annotation - FIX: anns is a list, so use anns[0]
    class_id = anns[0]['category_id']
    class_name = category_map[class_id]
    
    prompt = f"Kill: {class_name}"
    prompts.append({"image_id": image_id_str, "prompt": prompt})

with open("train_prompts_generated2.json", "w") as out_f:
    json.dump(prompts, out_f, indent=1)