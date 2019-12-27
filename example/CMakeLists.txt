cmake_minimum_required(VERSION 3.5)
project(example)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast")
include_directories(../include
                    ../onnx-tensorrt)

add_executable(buildEngine buildEngine.cpp)
target_link_libraries(buildEngine ctdet)

add_executable(runDet runDet.cpp)
target_link_libraries(runDet ctdet)
