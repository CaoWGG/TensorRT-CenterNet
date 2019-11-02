cmake_minimum_required(VERSION 3.5)
project(ctdet_trt)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_BUILD_TYPE Debug)
find_package(CUDA REQUIRED)

set(TENSORRT_ROOT /usr/local/TensorRT-5.0.2.6)

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include/)

message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_PARSER nvparsers
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
        HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN}
        ${TENSORRT_LIBRARY_PARSER} ${TENSORRT_LIBRARY_ONNXPARSER})
message(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")

find_package_handle_standard_args(
        TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)

if(NOT TENSORRT_FOUND)
    message(ERROR
            "Cannot find TensorRT library.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast")

list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")
cuda_include_directories(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARIES_DIRS})

include_directories(include ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})

file(GLOB CPP_SRC src/*.cpp)
file(GLOB CU_SRC src/*.cu)

cuda_add_executable(buildEngine example/buildEngine.cpp ${CPP_SRC} ${CU_SRC})
target_link_libraries(buildEngine ${OpenCV_LIBS} ${TENSORRT_LIBRARY})

cuda_add_executable(runDet example/runDet.cpp ${CPP_SRC} ${CU_SRC})
target_link_libraries(runDet ${OpenCV_LIBS} ${TENSORRT_LIBRARY})