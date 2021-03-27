import json
import cv2

file = "../data/data_annotated/2_0.json"

with open(file, "r+") as f:
    data = json.load(f)
