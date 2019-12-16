# TensorRT-CenterNet
### demo (GT 1070)
* ![image](img/show.gif)
* ![image](img/show2.png)

### Performance
| backbone       | input_size | GPU      | mode   | inference Time |
|----------------|------------|----------|--------|---------------|
| [mobilenetv2](https://github.com/CaoWGG/Mobilenetv2-CenterNet)    | 512x512    | gtx 1070 |float32 |    3.798ms    |
| [mobilenetv2](https://github.com/CaoWGG/Mobilenetv2-CenterNet)   | 512x512    | jetson TX2|float16 |    22ms      | 

### Enviroments
1. gtx 1070
```
ubuntu 1604
TensorRT 5.0
```
2. jetson TX2
```
jetpack 4.2
```
### Models
1. Convert [CenterNet](https://github.com/xingyizhou/centernet) model to onnx (deform conv is not support)
2. Use [netron](https://github.com/lutzroeder/netron) to observe whether the output of the converted onnx model is (hm, reg, wh)
3. for centerface
```bash
# a simple way to change input shape of (model.onnx)
import onnx
input_size = (512,512)
model = onnx.load_model("centerface.onnx")
d = model.graph.input[0].type.tensor_type.shape.dim
rate = (input_size[0]//d[2].dim_value,input_size[1]//d[3].dim_value)
d[0].dim_value = 1
d[2].dim_value *= rate[0]
d[3].dim_value *= rate[1]
for output in model.graph.output:
    d = output.type.tensor_type.shape.dim
    d[0].dim_value = 1
    d[2].dim_value *= rate[0]
    d[3].dim_value *= rate[1]
onnx.save_model(model,"centerface_changed_shape.onnx")
```

### Example
```bash
git clone https://github.com/CaoWGG/TensorRT-CenterNet.git
cd TensorRT-CenterNet
mkdir build
cd build && cmake .. && make
cd ..

##cthelmet   | config include/ctdetConfig.h
./buildEngine -i model/ctdet_helmet.onnx -o model/ctdet_helmet.engine
./runDet -e model/ctdet_helmet.engine -i test.jpg -c test.h264

##centerface | config include/ctdetConfig.h 
./buildEngine -i model/centerface.onnx -o model/centerface.engine
./runDet -e model/centerface.engine -i test.jpg -c test.h264

## a simple python demo (need cv2 numpy)
python3 run.py
```

### Related projects
* [TensorRT-Yolov3](https://github.com/lewes6369/TensorRT-Yolov3)
* [CenterNet](https://github.com/xingyizhou/centernet)
* [centerface](https://github.com/Star-Clouds/centerface)
* [netron](https://github.com/lutzroeder/netron)
* [cpp-optparse](https://github.com/weisslj/cpp-optparse)
