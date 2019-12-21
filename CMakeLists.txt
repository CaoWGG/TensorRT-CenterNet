cmake_minimum_required(VERSION 3.5)
project(ctdet_trt)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_BUILD_TYPE Debug)
set(GPU_ARCHS 61)  ## config your GPU_ARCHS
set(TENSORRT_ROOT /usr/local/TensorRT-5.0.2.6)
add_subdirectory(onnx-tensorrt)

find_package(CUDA REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include/)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER}
         nvonnxparser nvonnxparser_runtime)

message(STATUS ${TENSORRT_LIBRARY})
find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARIES_DIRS})

include_directories(include ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR} onnx-tensorrt )

file(GLOB CPP_SRC src/*.cpp)
file(GLOB CU_SRC src/*.cu)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast")

list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")


cuda_add_library(ctdet SHARED src/python_api.cpp ${CPP_SRC} ${CU_SRC})
target_include_directories(ctdet PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
target_link_libraries(ctdet ${TENSORRT_LIBRARY} ${OpenCV_LIBS})


cuda_add_executable(buildEngine example/buildEngine.cpp ${CPP_SRC} ${CU_SRC})
target_link_libraries(buildEngine ${OpenCV_LIBS} ${TENSORRT_LIBRARY})

cuda_add_executable(runDet example/runDet.cpp ${CPP_SRC} ${CU_SRC})
target_link_libraries(runDet ${OpenCV_LIBS} ${TENSORRT_LIBRARY})