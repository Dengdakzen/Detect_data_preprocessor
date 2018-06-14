import cv2
import json

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

filename = '../Detect_data/Position/' + str(20) + '.json'
with open(filename,'r') as F:
    data = json.load(F)['Players']
    # print(data)
k = data[10]
j = data[10]

print(bb_intersection_over_union((k['x'],k['y'],k['x']+k['width'],k['y']+k['height']),(j['x'] + 20,j['y']+20,j['x']+20+j['width'],j['y']+20+j['height'])))