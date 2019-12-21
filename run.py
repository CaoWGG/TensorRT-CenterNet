from ctypes import c_float,c_int,Structure,POINTER,CDLL,RTLD_GLOBAL,c_char_p,c_void_p,c_long
import cv2
import numpy as np
import os

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

### you need config include/ctdetConfig.h classNum and visThresh
class_name = ['person','helmet']
set_device(0)
net = init_net(b'model/ctdet_helmet.engine')
vid = cv2.VideoCapture('test.h264')
cv2.namedWindow('',cv2.WINDOW_NORMAL)
cv2.resizeWindow('',1024,768)
while True:
    ret,frame = vid.read()
    if not ret:
        break
    img_h,img_w,_ = frame.shape
    img = img_preprocess(frame,(512,512),mean = (0.485,0.456,0.406),std =(0.229,0.224,0.225))
    img_p = ndarray_to_image(img)
    pred = inference(net,img_p,img_w,img_h)
    free(img_p)
    show_img(pred,frame)
    free_result(pred)
    cv2.imshow('',frame)
    if cv2.waitKey(1)& 0xff == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
