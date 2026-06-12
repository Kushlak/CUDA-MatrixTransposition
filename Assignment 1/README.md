# Assignment 1 - Matrix Transpose

This assignment benchmarks matrix transposition on CPU and CUDA GPU.

## Files

- `src/matrix_transpose.cu` - main CUDA source file.
- `results/timings.csv` - saved benchmark results.

## What is implemented

- CPU transpose baseline.
- Naive CUDA transpose kernel.
- Tiled CUDA transpose kernel with shared memory.
- Correctness checks comparing CPU and GPU results.
- Timing output and CSV benchmark results.

## Build and run on Windows

Open PowerShell or the x64 Native Tools Command Prompt for Visual Studio 2022.

```powershell
nvcc src/matrix_transpose.cu -o matrix_transpose.exe
.\matrix_transpose.exe
```

If `nvcc` cannot find `cl.exe`, open `x64 Native Tools Command Prompt for Visual Studio 2022` and run the same commands there.

## Short explanation

The CPU version is used as the correct reference result. The naive GPU version gives one CUDA thread one matrix element to transpose. The tiled version uses shared memory so each block works on a small tile of the matrix. The program checks correctness and prints timing results.
