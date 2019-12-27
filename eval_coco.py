from ctypes import c_float,c_int,Structure,POINTER,CDLL,RTLD_GLOBAL,c_char_p,c_void_p,c_long
import cv2
import numpy as np
import os
from pycocotools.cocoeval import COCOeval
import pycocotools.coco as coco
from tqdm import tqdm
import sys

class BOX(Structure):
    _fields_ = [("x1", c_float),
                ("y1", c_float),
                ("x2", c_float),
                ("y2", c_float)]

class landmarks(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classId", c_int),
                ("prob", c_float),
                ("marks", landmarks * 5)]

class detRESULT(Structure):
    _fields_ = [("num", c_int),
                ("det", POINTER(DETECTION))]


lib=CDLL("lib/libctdet.so",RTLD_GLOBAL)
init_net=lib.initNet
init_net.argtypes = [c_char_p]
init_net.restype = c_void_p

free=lib.free
free.argtypes = [c_void_p]

free_result=lib.freeResult
free_result.argtypes = [POINTER(detRESULT)]

free_net=lib.freeNet
free_net.argtypes = [c_void_p]

inference=lib.predict
inference.argtypes=[c_void_p,c_void_p,c_int,c_int]
inference.restype=detRESULT

ndarray_image = lib.ndarrayToImage
ndarray_image.argtypes = [POINTER(c_float), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = c_void_p

set_device = lib.setDevice
set_device.argtypes = [c_int]


def ndarray_to_image(img):
    img = img.astype(np.float32)
    data = img.ctypes.data_as(POINTER(c_float))
    image_p = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image_p

def img_preprocess(image, target_shape=(512,512),mean = (0.485,0.456,0.406),std =(0.229,0.224,0.225)):
    mean = np.array(mean,dtype=np.float32).reshape([1,1,3])
    std = np.array(std,dtype=np.float32).reshape([1,1,3])
    h_target, w_target = target_shape
    h_org, w_org, _ = image.shape
    resize_ratio = min(1.0 * w_target / w_org, 1.0 * h_target / h_org)
    resize_w = int(resize_ratio * w_org)
    resize_h = int(resize_ratio * h_org)
    image_resized = cv2.resize(image, (resize_w, resize_h))
    image_paded = np.full((h_target, w_target, 3), 0).astype(np.uint8)
    dw = int((w_target - resize_w) / 2)
    dh = int((h_target - resize_h) / 2)
    image_paded[dh:resize_h+dh, dw:resize_w+dw,:] = image_resized
    image_paded = (image_paded.astype(np.float32)/255. -  mean) / std
    return image_paded

def show_img(pred,image):
    image_h, image_w, _ = image.shape
    bbox_thick = int(1.0 * (image_h + image_w) / 600)
    for j in range(pred.num):
        x1, y1, x2, y2, classId, prob = pred.det[j].bbox.x1, pred.det[j].bbox.y1, \
                                    pred.det[j].bbox.x2, pred.det[j].bbox.y2, pred.det[j].classId, pred.det[j].prob
        cls=int(classId)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),(255,0,0), bbox_thick)
        cv2.putText(image, '%s  %f'%(class_name[cls],prob), (int(x2), int(y2)), 0, 0.001 * image_h, (0, 255, 0),
                    bbox_thick // 2)
    return image

def top_k(pred,K=100):
    boxs_list = []
    for j in range(pred.num):
        x1, y1, x2, y2, classId, prob = pred.det[j].bbox.x1, pred.det[j].bbox.y1, \
                                    pred.det[j].bbox.x2, pred.det[j].bbox.y2, pred.det[j].classId, pred.det[j].prob
        boxs_list.append([x1,y1,x2,y2,classId,prob])
    boxs_list = np.array(boxs_list)
    if len(boxs_list)>K:
        prob = boxs_list[:,-1]
        top_arg = np.argsort(prob)[::-1][:K]
        boxs_list = boxs_list[top_arg]
    return boxs_list
### you need config include/ctdetConfig.h classNum and visThresh
## for coco
class_name = [
       'person', 'bicycle', 'car', 'motorcycle', 'airplane',
      'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
      'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
      'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
      'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
      'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
      'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
      'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
      'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
      'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
      'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
      'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
      'scissors', 'teddy bear', 'hair drier', 'toothbrush']
valid_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]

coco_val_ann = '/data/DataSet/coco2017/annotations/instances_val2017.json'
coco_val_dir = '/data/DataSet/coco2017/val2017'
data = coco.COCO(coco_val_ann)

set_device(0)
net = init_net(bytes(sys.argv[1],encoding = "utf8"))
detections = []
for img_id in tqdm(data.getImgIds()):
    img_name = os.path.join(coco_val_dir,data.loadImgs(ids=[img_id])[0]['file_name']).strip()
    frame  = cv2.imread(img_name)
    img_h, img_w, _ = frame.shape
    img = img_preprocess(frame, (512, 512), mean=(0.408, 0.447, 0.470), std=(0.289, 0.274, 0.278))
    img_p = ndarray_to_image(img)
    pred = inference(net,img_p,img_w,img_h)
    boxs = top_k(pred)
    for i,det in enumerate(boxs):
        x, y, x1, y1, cls, conf = det[:6]
        detection = {
            "image_id": img_id,
            "category_id": int(valid_ids[int(cls)]),
            "bbox": [x, y, x1 - x, y1 - y],
            "score": float("{:.2f}".format(conf))
        }
        detections.append(detection)

    free_result(pred)
    free(img_p)

coco_dets = data.loadRes(detections)
coco_eval = COCOeval(data, coco_dets, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
