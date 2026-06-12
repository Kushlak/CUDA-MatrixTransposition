# Assignment 2 - Image Convolution

This assignment benchmarks 2D image convolution on CPU and CUDA GPU.

## Files

- `src/convolution_filter.cu` - main CUDA source file.
- `results/timings.csv` - saved benchmark results.

## What is implemented

- CPU convolution baseline.
- Naive CUDA convolution.
- Shared memory CUDA convolution.
- Constant memory for filter values.
- Separable filter version.
- Correctness checks against the CPU result.
- Benchmark results for different image and filter sizes.

## Build and run on Windows

Open PowerShell or the x64 Native Tools Command Prompt for Visual Studio 2022.

```powershell
nvcc src/convolution_filter.cu -o convolution_filter.exe
.\convolution_filter.exe
```

If `nvcc` cannot find `cl.exe`, open `x64 Native Tools Command Prompt for Visual Studio 2022` and run the same commands there.

## Short explanation

The CPU result is used as the reference. Each GPU version computes the same convolution and then compares its output with the CPU output. Shared memory reduces repeated global memory reads. Constant memory is used for small filter values because every thread reads the same filter.
