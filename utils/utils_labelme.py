# utils_labelme.py
import json
import numpy as np

def polygon_to_bbox(points):
    pts = np.array(points, dtype=np.float32)
    x1 = float(pts[:,0].min()); y1 = float(pts[:,1].min())
    x2 = float(pts[:,0].max()); y2 = float(pts[:,1].max())
    return [x1, y1, x2, y2]

# usage
# j = json.load(open("image093515.json"))
# for shape in j['shapes']:
#    if shape['shape_type'] == 'polygon':
#       bbox = polygon_to_bbox(shape['points'])
