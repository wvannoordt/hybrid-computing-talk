#pragma once

#include <cuda.h>
#include "cuda_device_runtime_api.h"
#include "cuda_runtime_api.h"
#include <cuda_runtime.h>

#define _gt_hybrid __host__ __device__

#ifdef __NVCC__
#define _gt_cuda 1
#else
#define _gt_cuda 0
#endif