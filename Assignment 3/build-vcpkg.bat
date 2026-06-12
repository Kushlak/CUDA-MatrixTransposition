@echo off
setlocal

set "VS_VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "CMAKE_EXE=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA_EXE=%CD%\external\vcpkg\downloads\tools\ninja-1.13.2-windows\ninja.exe"
set "VCPKG_TOOLCHAIN=%CD%\external\vcpkg\scripts\buildsystems\vcpkg.cmake"
set "CUDA_NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe"

call "%VS_VCVARS%"
if errorlevel 1 exit /b 1

"%CMAKE_EXE%" -S . -B build-vcpkg -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="%NINJA_EXE%" ^
  -DCMAKE_TOOLCHAIN_FILE="%VCPKG_TOOLCHAIN%" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_CUDA_COMPILER="%CUDA_NVCC%"
if errorlevel 1 exit /b 1

"%CMAKE_EXE%" --build build-vcpkg --config Release
if errorlevel 1 exit /b 1

endlocal
