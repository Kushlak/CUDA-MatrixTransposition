# Assignment 3 - HDR Tone Mapping

This assignment applies HDR tone mapping to webcam frames and compares CPU and GPU processing.

## Files

- `src/main.cpp` - application entry point and webcam loop.
- `src/hdr_cpu.cpp` - CPU HDR tone mapping.
- `src/hdr_cuda.cu` - CUDA HDR tone mapping.
- `src/filter_utils.cpp` - filter and argument helper functions.
- `include/` - project headers.
- `CMakeLists.txt` - CMake build file.
- `build-vcpkg.bat` - helper script for a vcpkg-based build.

## Requirements

- CUDA Toolkit.
- Visual Studio C++ build tools.
- OpenCV available to CMake.

## Build and run on Windows

Open `x64 Native Tools Command Prompt for Visual Studio 2022`.

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\Release\cuda-webcam-filter.exe --filter hdr_tonemapping --compare
```

If the executable is placed directly in `build`, run:

```powershell
.\build\cuda-webcam-filter.exe --filter hdr_tonemapping --compare
```

## Useful options

```powershell
.\build\Release\cuda-webcam-filter.exe --filter hdr_tonemapping --exposure 1.2 --gamma 2.2 --saturation 1.1 --tone-operator reinhard
.\build\Release\cuda-webcam-filter.exe --filter hdr_tonemapping --cpu
.\build\Release\cuda-webcam-filter.exe --filter grayscale
```

## Short explanation

The program reads frames from the webcam. The CPU version is used as a reference, and the GPU version applies the same HDR tone mapping with CUDA. The compare mode prints CPU and GPU timings so the speed difference can be explained.
