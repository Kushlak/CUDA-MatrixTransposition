cmake_minimum_required(VERSION 3.18)

project(cuda_webcam_hdr_tonemapping LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(cuda-webcam-filter
    src/main.cpp
    src/filter_utils.cpp
    src/hdr_cpu.cpp
    src/hdr_cuda.cu
)

target_include_directories(cuda-webcam-filter PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cuda-webcam-filter PRIVATE ${OpenCV_LIBS})

set_target_properties(cuda-webcam-filter PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
