cmake_minimum_required (VERSION 3.0)
project (MCrec)
find_package(CUDA)

cuda_add_executable(mcrec mcrec.cu)

set_property(TARGET mcrec PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# The version number.
set (Tutorial_VERSION_MAJOR 0)
set (Tutorial_VERSION_MINOR 0)


