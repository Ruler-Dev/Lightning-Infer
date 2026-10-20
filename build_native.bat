@echo off
setlocal
REM Native MSVC (VS 2022 Build Tools) + CUDA 13.3 Windows build driver for Infernix.
REM Based on PR #13 (docs/windows.md) with one local adaptation: dependencies come
REM from the prebuilt global vcpkg tree at C:\vcpkg (ffmpeg 9.0.1 + curl 8.21
REM x64-windows) instead of manifest-mode rebuilding (ffmpeg 8.1.1 source build
REM fails here). No vcpkg toolchain file: deps are found via CMAKE_PREFIX_PATH
REM plus the local FindFFMPEG module.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
REM vcvars64 exports its own VCPKG_ROOT (the VS-bundled tree); re-assert ours.
set "VCPKG_ROOT=C:\vcpkg"
set "VCPKG_TARGET_TRIPLET=x64-windows"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
cd /d "%~dp0"
if "%1"=="configure" (
  cmake -S . -B build-windows -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DVCPKG_TARGET_TRIPLET=x64-windows ^
    -DCMAKE_PREFIX_PATH="C:\vcpkg\installed\x64-windows" ^
    -DCMAKE_CUDA_ARCHITECTURES=120a ^
    -DINFERNIX_BUILD_APPS=ON ^
    -DBUILD_TESTING=ON
) else if "%1"=="build" (
  cmake --build build-windows --config Release --parallel
) else (
  echo usage: build_native.bat [configure|build]
  exit /b 2
)
exit /b %errorlevel%
