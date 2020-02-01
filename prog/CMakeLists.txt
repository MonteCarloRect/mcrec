cmake_minimum_required(VERSION 2.8)
project(mcrec)
find_package(CUDA QUIET REQUIRED)
#find_package(C)
find_library(CURAND_LIB curand)

#CUDA_ADD_LIBRARY(curand)

include_directories(
    initial
    write
    rect
)

cuda_add_executable(
    mcrec
    global.h
    mcrec.h
    initial.h
    mcrec.cu
    initial/device_prop.cu
    initial/read_options.cu
    initial/read_gro.cu
    initial/initial_flows.cu
    initial/free_arrays.cu
    write/log.cu
    initial/read_top.cu
    initial/single_box.cu
    initial/data_to_device.cu
    initial/data_from_device.cu
    initial/rcut.cu
    write/single_box_write.cu
    rect/double_box.cu
    rect/dbox_init.cu
    rect/double_devtohost.cu
    rect/double_step.cu
)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -o3")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lm")

set_property(TARGET mcrec PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(mcrec m)

#target_link_libraries(
#    hellocuda
#) cmake .. -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -DCMAKE_CXX_COMPILER=/usr/bin/g++-6
