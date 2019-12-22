//
// Created by cao on 19-10-26.
//

#ifndef CTDET_TRT_CTDETCONFIG_H
#define CTDET_TRT_CTDETCONFIG_H

namespace ctdet{

    constexpr static float visThresh = 0.3;
    constexpr static int kernelSize = 3 ;  /// nms maxpool size


    //ctdet  ctdet_coco_dla_2x.onnx
    constexpr static int input_w = 512 ;
    constexpr static int input_h = 512 ;
    constexpr static int channel = 3 ;
    constexpr static int classNum = 80 ;
    constexpr static float mean[]= {0.408, 0.447, 0.470};
    constexpr static float std[] = {0.289, 0.274, 0.278};
    constexpr static char *className[]= {(char*)"person", (char*)"bicycle", (char*)"car", (char*)"motorcycle",
                                         (char*)"airplane", (char*)"bus", (char*)"train", (char*)"truck", (char*)"boat",
                                         (char*)"traffic light", (char*)"fire hydrant", (char*)"stop sign",
                                         (char*)"parking meter", (char*)"bench", (char*)"bird", (char*)"cat",
                                         (char*)"dog", (char*)"horse", (char*)"sheep", (char*)"cow", (char*)"elephant",
                                         (char*)"bear", (char*)"zebra", (char*)"giraffe", (char*)"backpack",
                                         (char*)"umbrella", (char*)"handbag", (char*)"tie", (char*)"suitcase",
                                         (char*)"frisbee", (char*)"skis", (char*)"snowboard", (char*)"sports ball",
                                         (char*)"kite", (char*)"baseball bat", (char*)"baseball glove", (char*)"skateboard",
                                         (char*)"surfboard", (char*)"tennis racket", (char*)"bottle", (char*)"wine glass",
                                         (char*)"cup", (char*)"fork", (char*)"knife", (char*)"spoon", (char*)"bowl",
                                         (char*)"banana", (char*)"apple", (char*)"sandwich", (char*)"orange",
                                         (char*)"broccoli", (char*)"carrot", (char*)"hot dog", (char*)"pizza",
                                         (char*)"donut", (char*)"cake", (char*)"chair", (char*)"couch",
                                         (char*)"potted plant", (char*)"bed", (char*)"dining table",
                                         (char*)"toilet", (char*)"tv", (char*)"laptop", (char*)"mouse",
                                         (char*)"remote", (char*)"keyboard", (char*)"cell phone",
                                         (char*)"microwave", (char*)"oven", (char*)"toaster", (char*)"sink",
                                         (char*)"refrigerator", (char*)"book", (char*)"clock", (char*)"vase",
                                         (char*)"scissors", (char*)"teddy bear", (char*)"hair drier", (char*)"toothbrush"};

/*
    //cthelmet
    constexpr static int input_w = 512 ;
    constexpr static int input_h = 512 ;
    constexpr static int channel = 3 ;
    constexpr static int classNum = 2 ;
    constexpr static float mean[]= {0.485,0.456,0.406};
    constexpr static float std[] = {0.229,0.224,0.225};
    constexpr static char *className[]= {(char*)"person",(char*)"helmet"};
*/

/*
    //ctface
    constexpr static int input_w = 1920 ;
    constexpr static int input_h = 1056 ;
    constexpr static int channel = 3 ;
    constexpr static int classNum = 1 ;
    constexpr static float mean[]= {0,0,0};
    constexpr static float std[] = {1,1,1};
    constexpr static char *className[]= {(char*)"face"};
*/
}
#endif //CTDET_TRT_CTDETCONFIG_H
