# Assignment 4 - CUDA Filter Pipeline

This assignment runs a real-time CUDA image filter pipeline and includes benchmark output.

## Files

- `src/main.cpp` - application entry point, benchmark mode, and UI loop.
- `src/kernels/convolution_kernels.cu` - CUDA filter kernels.
- `src/input_args_parser/` - command-line argument parsing.
- `src/utils/` - input and filter helper functions.
- `tests/unit_tests/` - unit tests.
- `pipeline_benchmark.csv` - benchmark table.
- `pipeline_benchmark.png` - benchmark chart.
- `CMakeLists.txt` - CMake build file.

## Requirements

- CUDA Toolkit.
- Visual Studio C++ build tools.
- OpenCV.
- The external CMake dependencies expected by `CMakeLists.txt`: `cxxopts`, `plog`, and `gtest`.

## Build and run on Windows

Open `x64 Native Tools Command Prompt for Visual Studio 2022`.

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\Release\cuda-webcam-filter.exe --benchmark
```

If the executable is placed directly in `build`, run:

```powershell
.\build\cuda-webcam-filter.exe --benchmark
```

## Useful options

```powershell
.\build\Release\cuda-webcam-filter.exe --filter blur
.\build\Release\cuda-webcam-filter.exe --pipeline blur,sharpen,edge --multi-stream
.\build\Release\cuda-webcam-filter.exe --transition --transition-from blur --transition-to sharpen
.\build\Release\cuda-webcam-filter.exe --benchmark --benchmark-frames 120
```

## Short explanation

The program applies one or more filters to frames. The pipeline version can run several filters in sequence, and the multi-stream option uses CUDA streams for more overlap. Benchmark mode creates synthetic frames, runs the pipeline, and writes timing results to CSV and PNG files.
