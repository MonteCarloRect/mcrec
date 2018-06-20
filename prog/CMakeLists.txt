cmake_minimum_required(VERSION 2.8)
project(mcrec)
find_package(CUDA QUIET REQUIRED)



include_directories(
    initial
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
    write/log.cu
)

set_property(TARGET mcrec PROPERTY CUDA_SEPARABLE_COMPILATION ON)

#target_link_libraries(
#    hellocuda
#) cmake .. -DCMAKE_C_COMPILER=/usr/bin/gcc-6
