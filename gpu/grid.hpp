#pragma once

#include "cuda_stuff.h"

#include <vector>
#include <iostream>

#include "complex.hpp"

template <typename data_t>
struct grid_t
{
    std::vector<data_t> raw;
    data_t* d_raw;
    std::size_t nx, ny;
    double xmin, xmax, ymin, ymax;
    grid_t (
        const data_t& fill,
        std::size_t nx_in,
        std::size_t ny_in,
        double xmin_in,
        double xmax_in,
        double ymin_in,
        double ymax_in)
    : nx{nx_in}, ny{ny_in}, xmin{xmin_in}, xmax{xmax_in}, ymin{ymin_in}, ymax{ymax_in}
    {
        raw.resize(nx*ny, fill);
        cudaMalloc(&d_raw, raw.size()*sizeof(data_t));
        cudaMemcpy(d_raw, &raw[0], raw.size()*sizeof(data_t), cudaMemcpyHostToDevice);
    }
    
    ~grid_t()
    {
        cudaFree(d_raw);
    }
};