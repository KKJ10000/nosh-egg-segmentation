import glob, os, re, json
import numpy as np, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def list_images(root=os.path.join(ROOT, "data/raw")):
    return sorted(glob.glob(os.path.join(root, "*", "*", "*.jpg")))

def stem(f):
    return os.path.splitext(os.path.basename(f))[0]

def source_id(f):
    """Roboflow renames 'not_damaged_12.jpg' -> 'not_damaged_12_jpg.rf.<hash>.jpg'; recover the original name."""
    m = re.match(r"(.+?)_jpg\.rf\.", os.path.basename(f))
    return m.group(1) if m else stem(f)

def load_split():
    with open(os.path.join(ROOT, "results/split.json")) as fh:
        return json.load(fh)

def load_gt(st):
    """SAM-derived instance label map (uint16, 0 = background)."""
    return cv2.imread(os.path.join(ROOT, "outputs/sam_masks", st + ".png"), cv2.IMREAD_UNCHANGED)
