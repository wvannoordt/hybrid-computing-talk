#pragma once

#include "hybrid/device.hpp"

namespace hybrid
{
#if (_gt_cuda)
    template <typename krange_t, typename kernel_t>
    __global__ void k_exec(krange_t range, kernel_t kernel)
    {
        const int ij[2] = {threadIdx.x + blockIdx.x*blockDim.x, threadIdx.y + blockIdx.y*blockDim.y};
        if ((ij[0] < range.nx) && (ij[1] < range.ny))
        {
            kernel(ij);
        }
    }
#endif
    template <typename krange_t, typename kernel_t, typename device_t>
    static void execute(const krange_t& range, kernel_t kernel, const device_t&)
    {
        if constexpr (!(is_gpu<device_t>))
        {
            // Can replace with a generic multi-dimensional loop
            for (int j = 0; j < range.ny; ++j)
            {
                for (int i = 0; i < range.nx; ++i)
                {
                    int ij[2] = {i,j};
                    kernel(ij);
                }
            }
        }
        else
        {
#if (_gt_cuda)
            // Can replace with a call to a kernel parameters function
            constexpr static int blk_wid = 8;
            dim3 grd_size, blk_size;
            grd_size.x = 1+(range.nx-1)/blk_wid;
            grd_size.y = 1+(range.ny-1)/blk_wid;
            grd_size.z = 1;
            blk_size.x = blk_wid;
            blk_size.y = blk_wid;
            blk_size.z = 1;
            k_exec<<<grd_size, blk_size>>>(range, kernel);
            cudaDeviceSynchronize();
#endif
        }
    }
}