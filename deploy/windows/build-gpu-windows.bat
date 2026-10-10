@echo off
rem Builds the node and the tools with the CUDA kernels on Windows (needs: Visual Studio 2022 Build Tools with C++, CUDA Toolkit 13.x, Rust).
rem Output: %CARGO_TARGET_DIR%\release (default C:\vt\release): node.exe, vearl-cli.exe, vearl-wallet.exe, vearl-genesis.exe, vearl-faucet.exe, gpubench.exe.
rem Windows cannot replace a node.exe that is running: build into another folder (set CARGO_TARGET_DIR=C:\vt2) before calling this script.
rem VEARL_CUDA_ARCH: 89 = RTX 40xx, 86 = RTX 30xx, 120 = RTX 50xx (a fast build for one card). To build a binary for every card (80,86,89,90,120 + PTX,
rem what the release uses) set VEARL_CUDA_ARCH=all before calling it.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "PATH=%CUDA_PATH%\bin;%USERPROFILE%\.cargo\bin;%PATH%"
if "%CARGO_TARGET_DIR%"=="" set "CARGO_TARGET_DIR=C:\vt"
if "%VEARL_CUDA_ARCH%"=="" set "VEARL_CUDA_ARCH=89"
if "%VEARL_CUDA_ARCH%"=="all" set "VEARL_CUDA_ARCH="
cd /d "%~dp0\..\..\vearl-core"
cargo build --release --features gpu --bin node --bin vearl-cli --bin vearl-wallet --bin vearl-genesis --bin vearl-faucet --bin gpubench
