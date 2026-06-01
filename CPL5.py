# Generated from: CPL5.ipynb
# Converted at: 2026-06-01T10:54:50.797Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

!pip install numba

import numpy as np
from numba import cuda

print("CUDA Available:", cuda.is_available())

if cuda.is_available():
    print("GPU Name:", cuda.get_current_device().name)

@cuda.jit
def bitonic_sort(arr, j, k):
    i = cuda.threadIdx.x
    ixj = i ^ j

    if ixj > i:
        # Ascending phase
        if (i & k) == 0:
            if arr[i] > arr[ixj]:
                temp = arr[i]
                arr[i] = arr[ixj]
                arr[ixj] = temp
        # Descending phase
        else:
            if arr[i] < arr[ixj]:
                temp = arr[i]
                arr[i] = arr[ixj]
                arr[ixj] = temp

N = 8

h_arr = np.array([1, 3, 5, 9, 8, 7, 4, 2], dtype=np.int32)

print("Original Array:", h_arr)

d_arr = cuda.to_device(h_arr)

threads_per_block = N
blocks = 1

k = 2
while k <= N:
    j = k // 2
    while j > 0:
        bitonic_sort[blocks, threads_per_block](d_arr, j, k)
        cuda.synchronize()
        j //= 2
    k *= 2

sorted_arr = d_arr.copy_to_host()

print("Sorted Array:", sorted_arr)