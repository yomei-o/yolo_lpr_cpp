#!/bin/sh
# CUDA build. The engine's only device seam is pure/backend.hpp (bk::gemm*_hosted), so the *same*
# sources build for CPU and GPU — this script only sets the toolchain up.
#
#   sh build/nvcc.sh pure/gpu_check.cpp -o gpu_check.exe
#   EXTRA="-DUSE_CUBLAS -lcublas" sh build/nvcc.sh ...      # cuBLAS instead of the hand kernel
#
# Windows notes (this machine): nvcc drives cl.exe, so it needs the same INCLUDE/LIB/PATH that
# build/cc.sh sets by hand (vcvarsall hangs here). C++17, not 20: nvcc 13.x does not accept
# -std=c++20 for device compilation.
set -e

VSROOT="/c/Program Files/Microsoft Visual Studio/2022"
MSVC_DIR=$(ls -d "$VSROOT"/*/VC/Tools/MSVC/* 2>/dev/null | sort -V | tail -1)
SDK_INC=$(ls -d "/c/Program Files (x86)/Windows Kits/10/Include"/* 2>/dev/null | sort -V | tail -1)
SDK_LIB=$(ls -d "/c/Program Files (x86)/Windows Kits/10/Lib"/* 2>/dev/null | sort -V | tail -1)
CUDA_DIR="${CUDA_DIR:-$(ls -d "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA"/* 2>/dev/null | sort -V | tail -1)}"
ARCH="${ARCH:-sm_75}"        # T4 / Colab default; sm_86 for A10/3090, sm_89 for L4/4090

SRC="$1"; shift
OUT="gpu_check.exe"
if [ "$1" = "-o" ]; then OUT="$2"; shift 2; fi

if [ -n "$MSVC_DIR" ] && [ -n "$SDK_INC" ]; then      # Windows: hand the host compiler its env
  BIN="$MSVC_DIR/bin/Hostx64/x64"
  export INCLUDE="$(cygpath -w "$MSVC_DIR/include");$(cygpath -w "$SDK_INC/ucrt");$(cygpath -w "$SDK_INC/um");$(cygpath -w "$SDK_INC/shared");$(cygpath -w "$SDK_INC/winrt")"
  export LIB="$(cygpath -w "$MSVC_DIR/lib/x64");$(cygpath -w "$SDK_LIB/ucrt/x64");$(cygpath -w "$SDK_LIB/um/x64")"
  export PATH="$BIN:$CUDA_DIR/bin:$PATH"
  NVCC="$CUDA_DIR/bin/nvcc.exe"
  # dash-form MSVC flags on purpose: Git Bash rewrites a leading slash into a Windows path
  HOSTFLAGS="-Xcompiler -utf-8 -Xcompiler -EHsc -Xcompiler -DNOMINMAX -Xcompiler -wd4819"
  LINK="-Xlinker shell32.lib"   # -Xlinker: nvcc would treat a bare .lib as a source file
else                                                  # Linux / Colab
  NVCC="${NVCC:-nvcc}"
  HOSTFLAGS=""
  LINK=""
fi

mkdir -p scratch
"$NVCC" -x cu -std=c++17 -O2 --extended-lambda -DUSE_CUDA -arch=$ARCH \
  -Ipure -Ipure/third_party \
  $HOSTFLAGS $EXTRA "$@" "$SRC" $LINK -o "$OUT"
echo "built $OUT (nvcc $("$NVCC" --version | tail -1 | tr -d '\r'), arch $ARCH)"
