cmake_minimum_required(VERSION 3.12)
project(md_float16)

find_package(cuda)
set(CUDA_NVCC_FLAGS -arch=compute_61;-code=sm_61)
if(CUDA_FOUND)
	message(${CUDA_LIBRARIES})
	message(${CUDA_INCLUDE_DIRS})
else(CUDA_FOUND)
	message("Can't find cuda library")
endif(CUDA_FOUND)



add_subdirectory(source)