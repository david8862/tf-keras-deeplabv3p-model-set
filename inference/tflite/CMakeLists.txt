cmake_minimum_required(VERSION 3.5)
project(deeplabSegment LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_BUILD_TYPE "Release")

#### specify the build flag
SET(CMAKE_CXX_FLAGS  "-std=c++14 -O2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
SET(TARGET_PLAT "linux_x86_64" CACHE STRING INTERNAL)

set(DEEPLAB_SEGMENT_SRC
        deeplabSegment.cpp)

#set(TF_ROOT_PATH /mnt/d/Downloads/tensorflow)
link_directories("${TF_ROOT_PATH}/tensorflow/lite/tools/make/gen/${TARGET_PLAT}/lib/")

add_executable(deeplabSegment ${DEEPLAB_SEGMENT_SRC})

target_include_directories(deeplabSegment PRIVATE ${TF_ROOT_PATH} ${TF_ROOT_PATH}/tensorflow/lite/tools/make/downloads/flatbuffers/include)
target_link_libraries(deeplabSegment PRIVATE libtensorflow-lite.a -lstdc++ -lpthread -lm -ldl -lrt)
#target_link_libraries(deeplabSegment PRIVATE -ltensorflow-lite -lstdc++ -lpthread -lm -ldl -lrt)
