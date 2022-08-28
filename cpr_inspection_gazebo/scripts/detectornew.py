
from core.utils import decode_cfg, load_weights
from core.image import draw_bboxes, preprocess_image, postprocess_image, read_image, read_video, Shader
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import tensorflow as tf
import sys
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
import os
curr_path = os.path.dirname(os.path.abspath(__file__))
from headers import YoloV4Header as Header
from core.model.one_stage.yolov4 import YOLOv4_Tiny as Model

cfg = decode_cfg(curr_path+"/cfgs/coco_yolov4_tiny.yaml")
model,evalmodel = Model(cfg,416)

init_weight_path = cfg['train']['init_weight_path']
if init_weight_path:
    print('Load Weights File From:', init_weight_path)
    load_weights(model, init_weight_path)
else:
    raise SystemExit('init_weight_path is Empty !')



shader = Shader(cfg['yolo']['num_classes'])
names = cfg['yolo']['names']
# image_size = cfg['test']['image_size'][0]
image_size = 416

iou_threshold = cfg["yolo"]["iou_threshold"]
score_threshold = cfg["yolo"]["score_threshold"]
max_outputs = cfg["yolo"]["max_boxes"]
num_classes = cfg["yolo"]["num_classes"]
strides = cfg["yolo"]["strides"]
mask = cfg["yolo"]["mask"]
anchors = cfg["yolo"]["anchors"]




print(image_size)

def preprocess_image(image, size, bboxes=None):
    """
    :param image: RGB, uint8
    :param size:
    :param bboxes:
    :return: RGB, uint8
    """
    iw, ih = size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], dtype=np.uint8, fill_value=127)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized

    if bboxes is None:
        return image_paded

    else:
        bboxes = np.asarray(bboxes).astype(np.float32)
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + dw
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + dh

        return image_paded, bboxes




def inference(image):
    h, w = image.shape[:2]
    image = preprocess_image(image, (image_size, image_size)).astype(np.float32)
    images = np.expand_dims(image, axis=0)
    images  = images/255.

    tic = time.time()
    pred = model.predict(images)
    bboxes, scores, classes, valid_detections = Header(80, anchors, mask, strides, 10,
                  iou_threshold, score_threshold,inputs = pred)

    # bboxes, scores, classes, valid_detections = evalmodel.predict(images)

    toc = time.time()

    bboxes = bboxes[0][:valid_detections[0]]
    scores = scores[0][:valid_detections[0]]
    classes = classes[0][:valid_detections[0]]
    _, bboxes = postprocess_image(image, (w, h), bboxes.numpy())

    return (toc - tic) * 1000, bboxes, scores, classes


# w=640
# h=480

# cap = cv2.VideoCapture(0)
# start = time.time()
# while(True):
#     ret, frame = cap.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     ms, bboxes, scores, classes = inference(frame)
#     image = draw_bboxes(frame, bboxes, scores, classes, names, shader)
#     cv2.imshow("image", image)
#     print('Inference Time:', ms, 'ms')
#     print('Fps:', 1000/ms)
#     frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()

# cv2.destroyAllWindows()

def run_model(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ms, bboxes, scores, classes = inference(frame)
    image = draw_bboxes(frame, bboxes, scores, classes, names, shader)
    # cv2.imshow("image", image)
    print('Inference Time:', ms, 'ms')
    print('Fps:', 1000/ms)
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow('frame',frame)
    return frame
