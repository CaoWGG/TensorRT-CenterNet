//
// Created by cao on 19-10-26.
//

#ifndef CTDET_TRT_CTDETCONFIG_H
#define CTDET_TRT_CTDETCONFIG_H

namespace ctdet{



     static float visThresh = 0.3;

     static int inputSize = 512 ;
     static int channel = 3 ;
     static int ouputSize = inputSize/4 ;
     static int kernelSize = 4 ;


    //cthelmet
     static int classNum = 2 ;
     static float mean[]= {0.485,0.456,0.406};
     static float std[] = {0.229,0.224,0.225};
     static char *className[]= {(char*)"person",(char*)"helmet"};
    //ctface
//     static int classNum = 1 ;
//     static float mean[]= {0,0,0};
//     static float std[] = {1,1,1};
//     static char *className[]= {(char*)"face"};

}
#endif //CTDET_TRT_CTDETCONFIG_H
