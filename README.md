# TensorRT-CenterNet
### demo (GT 1070)
* ![image](img/show.gif)
* ![image](img/show2.png)

### Performance
| backbone       | input_size | GPU      | mode   | inference Time |
|----------------|------------|----------|--------|---------------|
| [mobilenetv2](https://github.com/CaoWGG/Mobilenetv2-CenterNet)    | 512x512    | gtx 1070 |float32 |    3.798ms    |
| [mobilenetv2](https://github.com/CaoWGG/Mobilenetv2-CenterNet)    | 512x512    | gtx 1070 |int8    |    1.75ms    |
| [mobilenetv2](https://github.com/CaoWGG/Mobilenetv2-CenterNet)   | 512x512    | jetson TX2|float16 |    22ms      |
| [dla34](https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/pose_dla_dcn.py)       | 512x512    | gtx 1070 |float32 |    24ms    |
| [dla34](https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/pose_dla_dcn.py)       | 512x512    | gtx 1070 |int8    |    19.6ms    |
1. support deform conv v2.
2. no nms.
3. support fp32 fp16 int8 mode.

### Enviroments
1. gtx 1070
```
ubuntu 1604
TensorRT 5.0
onnx-tensorrt v5.0
```
2. jetson TX2
```
jetpack 4.2
```

### Models
1. Convert [CenterNet](https://github.com/xingyizhou/centernet) model to onnx.
2. Use [netron](https://github.com/lutzroeder/netron) to observe whether the output of the converted onnx model is (hm, reg, wh)

### Example
```bash
git clone https://github.com/CaoWGG/TensorRT-CenterNet.git
cd TensorRT-CenterNet
mkdir build
cd build && cmake .. && make
cd ..

##ctdet | config include/ctdetConfig.h 
## int 8
./buildEngine -i model/ctdet_coco_dla_2x.onnx -o model/ctdet_coco_dla_2x.engine -m 2 calib_img_list.txt
./runDet -e model/ctdet_coco_dla_2x.engine -i test.jpg -c test.h264

##cthelmet   | config include/ctdetConfig.h
## flaot32
./buildEngine -i model/ctdet_helmet.onnx -o model/ctdet_helmet.engine -m 0
./runDet -e model/ctdet_helmet.engine -i test.jpg -c test.h264
## int8
./buildEngine -i model/ctdet_helmet.onnx -o model/ctdet_helmet.engine -m 2 -c calib_img_list.txt
./runDet -e model/ctdet_helmet.engine -i test.jpg -c test.h264

##centerface | config include/ctdetConfig.h 
./buildEngine -i model/centerface.onnx -o model/centerface.engine
./runDet -e model/centerface.engine -i test.jpg -c test.h264

## a simple python demo (need cv2 numpy)
python3 run.py
```

### Related projects
* [TensorRT-Yolov3](https://github.com/lewes6369/TensorRT-Yolov3)
* [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt)
* [TensorRT](https://github.com/NVIDIA/TensorRT)
* [CenterNet](https://github.com/xingyizhou/centernet)
* [centerface](https://github.com/Star-Clouds/centerface)
* [netron](https://github.com/lutzroeder/netron)
* [cpp-optparse](https://github.com/weisslj/cpp-optparse)
