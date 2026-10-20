import glob
import os

import cv2

for p in sorted(glob.glob("shared/reference/*.jpg")):
    img = cv2.imread(p)
    h, w = img.shape[:2]
    s = 200 / max(h, w)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    print(f"{os.path.basename(p):12} sharpness {cv2.Laplacian(g, cv2.CV_64F).var():7.1f}   {w}x{h}")