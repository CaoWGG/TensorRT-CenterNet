# TensorRT-CenterNet
### demo (GT 1070)
![image](img/show.gif)

### Performance
| backbone       | input_size | GPU      | mode   | inference Time |
|----------------|------------|----------|--------|---------------|
| mobilenetv2    | 512x512    | gtx 1070 |float32 |    3.798ms    |
| mobilenetv2    | 512x512    | jetson TX2|float16 |    22ms      | 

### enviroments
1. gtx 1070
```
ubuntu 1604
TensorRT 5.0

```
2. jetson TX2
```
jetpack 4.2
```
### models
1. Convert [CenterNet] (https://github.com/xingyizhou/centernet) model to onnx (deform cov is not supported)
2. Use [netron](https://github.com/lutzroeder/netron) to observe whether the output of the converted onnx model is (hm, reg, wh)

### example
```bash
git clone https://github.com/CaoWGG/TensorRT-CenterNet.git
cd TensorRT-CenterNet
mkdir build
cd build && cmake .. && make
cd ..
./buildEngine -i model/ctdet_helmet.onnx -o model/ctdet_helmet.engine
./runDet -i model/ctdet_helmet.engine -img 000138.jpg -cap test/h264
```

### related projects
* [TensorRT-Yolov3](https://github.com/lewes6369/TensorRT-Yolov3)
* [CenterNet](centernet)
* [netron](https://github.com/lutzroeder/netron)
* [cpp-optparse](https://github.com/weisslj/cpp-optparse)