cmake_minimum_required(VERSION 3.5)
project(deeplabSegment LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_BUILD_TYPE "Release")

#### specify the compiler flag
SET(CMAKE_CXX_FLAGS  "-std=c++11 -O2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")

set(DEEPLAB_SEGMENT_SRC
        deeplabSegment.cpp)

#set(MNN_ROOT_PATH /mnt/d/Projects/MNN)

include_directories("${MNN_ROOT_PATH}/include/" "${MNN_ROOT_PATH}/3rd_party/imageHelper/")
link_directories("${MNN_ROOT_PATH}/build/")
add_executable(deeplabSegment ${DEEPLAB_SEGMENT_SRC})
target_link_libraries(deeplabSegment -lMNN -lstdc++ -lpthread)
#target_link_libraries(deeplabSegment libMNN.a -Wl,--whole-archive -Wl,--no-whole-archive -lstdc++ -lpthread)
