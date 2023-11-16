#pragma once

#include <vector>
#include <iostream>

#include "complex.hpp"

template <typename data_t>
struct grid_t
{
    std::vector<data_t> raw;
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
    }
    
    const data_t& operator () (const std::size_t i, const std::size_t j) const
    {
        return raw[i + nx*j];
    }
    
    data_t& operator () (const std::size_t i, const std::size_t j)
    {
        return raw[i + nx*j];
    }
    
    complex_t get_c(const std::size_t i, const std::size_t j) const
    {
        return {xmin + (double(i) + 0.5)*(xmax-xmin) / nx, ymin + (double(j) + 0.5)*(ymax-ymin) / ny};
    }
};