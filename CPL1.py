# Generated from: CPL1.ipynb
# Converted at: 2026-06-01T10:53:24.992Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

!nvidia-smi

from numba import cuda
import numpy as np


@cuda.jit
def hello_kernel():
  # Thread and block indices
  tx=cuda.threadIdx.x
  bx=cuda.blockIdx.x
  bdim=cuda.blockDim.x


  # Global thread ID
  gid = bx*bdim+tx
  print("hellp from block ",bx,"thread ",tx,"Global id ", gid)

blocks =2
threads_per_block=4
hello_kernel[blocks,threads_per_block]()
cuda.synchronize()