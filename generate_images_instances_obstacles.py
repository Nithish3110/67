import os, random, cv2, json, numpy as np

# -------------------- CONFIG --------------------
SPRITE_FOLDERS = ["pikachu_sprites","mewtwo_sprites",
                  "charizard_sprites","bulbasaur_sprites"]
OBSTACLE_FOLDER = "obstacles_colored"
BACKGROUNDS_FOLDER = "Backgrounds_normalized"
OUTPUT_DIR = "scenes"
JSON_PATH = "instances_generated.json"

NUM_IMAGES = 750
IMG_WIDTH, IMG_HEIGHT = 640, 480

# Pokémon
N_MEAN,N_STD,N_MIN,N_MAX = 6,1,4,8
SIZE_MEAN,SIZE_STD,SIZE_MIN,SIZE_MAX = 75,40,60,200
SPRITE_ROT_PROB,MAX_ROT_DEG = 0.6,30

# Sprite-attached obstacles
SPRITE_OBS_PROB = 0.7
SPRITE_OBS_ABOVE_PROB = 0.6
SPRITE_OBS_MIN,SPRITE_OBS_MAX = 1,2
SPRITE_OBS_SCALE = (0.3,0.7)

# Top-screen obstacles (rare)
TOP_OBS_PROB = 0.1
TOP_OBS_COUNT = (1,2)
TOP_OBS_WIDTH_FRAC = (0.3,0.8)
TOP_OBS_MAX_HEIGHT_FRAC = 0.5

# Obstacle appearance tweaks
OBSTACLE_GRAYSCALE_PROB = 0.5
OBSTACLE_FADE_PROB = 0.9       # probability of extra transparency
FADE_ALPHA_MIN,FADE_ALPHA_MAX = 0.4,0.8

MAX_POSITION_ATTEMPTS = 15
RANDOM_SEED = None
# --------------------------------------------------

# ---------- Helpers ----------
def sample_trunc_norm_int(mean,std,low,high):
    while True:
        v=np.random.normal(mean,std)
        if low<=v<=high: return int(round(v))

def list_imgs(folder):
    return [os.path.join(folder,f) for f in os.listdir(folder)
            if f.lower().endswith((".png",".jpg",".jpeg"))] if os.path.isdir(folder) else []

def overlay_alpha(bg,ov,x,y):
    h,w=ov.shape[:2]
    if y+h>bg.shape[0] or x+w>bg.shape[1]: return
    region=bg[y:y+h,x:x+w]
    if ov.shape[2]==4:
        alpha=(ov[:,:,3:4].astype(float)/255.0)
        region[:,:,:3]=(ov[:,:,:3]*alpha+region[:,:,:3]*(1-alpha)).astype(np.uint8)
    else: region[:,:,:3]=ov[:,:,:3]
    bg[y:y+h,x:x+w]=region

def overlap(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    return not(ax+aw<=bx or bx+bw<=ax or ay+ah<=by or by+bh<=ay)

def rotate(img,ang):
    h,w=img.shape[:2]; M=cv2.getRotationMatrix2D((w//2,h//2),ang,1)
    cos,sin=abs(M[0,0]),abs(M[0,1])
    nw,nh=int(h*sin+w*cos),int(h*cos+w*sin)
    M[0,2]+=nw/2-w//2; M[1,2]+=nh/2-h//2
    return cv2.warpAffine(img,M,(nw,nh),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0,0,0,0))

def grayscale_alpha(img):
    rgb,alpha=img[:,:,:3],img[:,:,3]
    gray=np.dot(rgb,[0.299,0.587,0.114]).astype(np.uint8)
    return np.dstack([np.repeat(gray[...,None],3,axis=-1),alpha])

def maybe_fade(img):
    if img.shape[2]!=4: return img
    if random.random()<OBSTACLE_FADE_PROB:
        factor=random.uniform(FADE_ALPHA_MIN,FADE_ALPHA_MAX)
        img=img.copy()
        img[:,:,3]=(img[:,:,3].astype(float)*factor).clip(0,255).astype(np.uint8)
    return img

# ---------- Scene generator ----------
def create_scene(img_id, ann_start):
    bg=cv2.imread(random.choice(list_imgs(BACKGROUNDS_FOLDER)),cv2.IMREAD_UNCHANGED)
    bg=cv2.resize(bg,(IMG_WIDTH,IMG_HEIGHT))
    if bg.shape[2]==3: bg=cv2.cvtColor(bg,cv2.COLOR_BGR2BGRA)
    H,W=bg.shape[:2]
    placements=[]; sprite_boxes=[]; annotations=[]
    ann_id = ann_start

    sprites=sum([list_imgs(f) for f in SPRITE_FOLDERS],[])
    n_sprites=sample_trunc_norm_int(N_MEAN,N_STD,N_MIN,N_MAX)

    folder_to_category = {
        "pikachu_sprites": 1,
        "charizard_sprites": 2,
        "bulbasaur_sprites": 3,
        "mewtwo_sprites": 4
    }

    # ---- Place Pokémon ----
    for _ in range(n_sprites):
        sp_path=random.choice(sprites)
        sp=cv2.imread(sp_path,cv2.IMREAD_UNCHANGED)
        tw=sample_trunc_norm_int(SIZE_MEAN,SIZE_STD,SIZE_MIN,SIZE_MAX)
        th=int(sp.shape[0]*(tw/sp.shape[1]))
        sp=cv2.resize(sp,(tw,th))
        if random.random()<SPRITE_ROT_PROB:
            sp=rotate(sp,random.uniform(-MAX_ROT_DEG,MAX_ROT_DEG))
        tw,th=sp.shape[1],sp.shape[0]
        for _ in range(MAX_POSITION_ATTEMPTS):
            x,y=random.randint(0,W-tw),random.randint(0,H-th)
            if not any(overlap((x,y,tw,th),b) for b in sprite_boxes):
                sprite_boxes.append((x,y,tw,th))
                placements.append((sp,x,y,"sprite",False))
                # ---- Annotation ----
                folder_name = [f for f in SPRITE_FOLDERS if f in sp_path][0]
                annotations.append({
                    "id": ann_id+3044,
                    "image_id": img_id+500,
                    "category_id": folder_to_category[folder_name],
                    "bbox": [x, y, tw, th],
                    "area": tw*th,
                    "iscrowd": 0
                })
                ann_id += 1
                break

    obstacles=list_imgs(OBSTACLE_FOLDER)

    # ---- Sprite-attached obstacles ----
    for (sp,x,y,tag,_) in list(placements):
        if random.random()<SPRITE_OBS_PROB:
            for _ in range(random.randint(SPRITE_OBS_MIN,SPRITE_OBS_MAX)):
                ob=cv2.imread(random.choice(obstacles),cv2.IMREAD_UNCHANGED)
                if random.random()<OBSTACLE_GRAYSCALE_PROB: ob=grayscale_alpha(ob)
                ob=maybe_fade(ob)
                scale=random.uniform(*SPRITE_OBS_SCALE)
                ow,oh=int(sp.shape[1]*scale),int(sp.shape[0]*scale)
                ob=cv2.resize(ob,(ow,oh))
                ox=x+random.randint(0,max(1,sp.shape[1]-ow))
                oy=y+random.randint(0,max(1,sp.shape[0]-oh))
                above = random.random() < SPRITE_OBS_ABOVE_PROB
                placements.append((ob,ox,oy,"sprite_obstacle",above))


    # ---- Render ----
    out=bg.copy()
    for im,x,y,tag,above in [p for p in placements if not p[4]]:
        overlay_alpha(out,im,x,y)
    for im,x,y,tag,above in [p for p in placements if p[4]]:
        overlay_alpha(out,im,x,y)

    return out, annotations, ann_id

# ---------- Dataset generator ----------
def generate_dataset():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id":1,"name":"pikachu"},
            {"id":2,"name":"charizard"},
            {"id":3,"name":"bulbasaur"},
            {"id":4,"name":"mewtwo"}
        ]
    }

    ann_id = 0
    for i in range(NUM_IMAGES):
        img, anns, ann_id = create_scene(i, ann_id)
        fname = f"img_{i+500:05d}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, fname), img)
        data["images"].append({
            "id": i+500,
            "file_name": fname,
            "width": IMG_WIDTH,
            "height": IMG_HEIGHT
        })
        data["annotations"].extend(anns)
        print(f"Saved {fname}")

    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nCOCO annotations saved to {JSON_PATH}")

# ------------------ RUN ------------------
if __name__ == "__main__":
    generate_dataset()
