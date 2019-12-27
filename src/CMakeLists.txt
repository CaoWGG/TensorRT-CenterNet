cmake_minimum_required(VERSION 3.5)
project(ctdet_trt)
find_package(CUDA REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include/)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
message(STATUS "Found TensorRT libs ${TENSORRT_LIBRARY_INFER}")

find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARIES_DIRS})

file(GLOB CPP_SRC *.cpp)
file(GLOB CU_SRC *.cu)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast")
list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")

include_directories(${CUDA_INCLUDE_DIRS}
                    ${TENSORRT_INCLUDE_DIR}
                    ${OpenCV_INCLUDE_DIRS}
                    ../include
                    ../onnx-tensorrt)

cuda_add_library(ctdet SHARED ${CPP_SRC} ${CU_SRC})
target_link_libraries(ctdet
        ${TENSORRT_LIBRARY_INFER}
        ${OpenCV_LIBS}
        nvonnxparser
        nvonnxparser_runtime)
