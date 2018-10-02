cmake_minimum_required(VERSION 2.8)
project(mcrec)
find_package(CUDA QUIET REQUIRED)



include_directories(
    initial
    write
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
)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -o3")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lm")

set_property(TARGET mcrec PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(mcrec m)

#target_link_libraries(
#    hellocuda
#) cmake .. -DCMAKE_C_COMPILER=/usr/bin/gcc-6
