#pragma once

#include <vector>
#include <iostream>

#include "complex.hpp"

#include "hybrid/vector.hpp"

template <typename data_t, typename device_t, const bool is_image = false, const bool is_const_image = false>
struct grid_t
{
    using container_type     = hybrid::vector_t<data_t, device_t>;
    using noconst_image_type = hybrid::vector_image_t<data_t>;
    using const_image_type   = hybrid::const_vector_image_t<data_t>;
    using image_type         = typename std::conditional<is_const_image, const_image_type, noconst_image_type>::type;
    using data_member_type   = typename std::conditional<is_image, image_type, container_type>::type;
    
    using self_image_type       = grid_t<data_t, device_t, true, false>;
    using self_const_image_type = grid_t<data_t, device_t, true, true>;
    
    std::size_t nx, ny;
    double xmin, xmax, ymin, ymax;
    
    data_member_type raw;
    
    grid_t() {}
    
    grid_t (
        const data_t& fill,
        std::size_t nx_in,
        std::size_t ny_in,
        double xmin_in,
        double xmax_in,
        double ymin_in,
        double ymax_in,
        const device_t&)
    : nx{nx_in}, ny{ny_in}, xmin{xmin_in}, xmax{xmax_in}, ymin{ymin_in}, ymax{ymax_in}
    {
        raw.resize(nx*ny);
    }
    
    _gt_hybrid const data_t& operator () (const std::size_t i, const std::size_t j) const
    {
        return raw[i + nx*j];
    }
    
    _gt_hybrid data_t& operator () (const std::size_t i, const std::size_t j)
    {
        return raw[i + nx*j];
    }
    
    _gt_hybrid complex_t get_c(const std::size_t i, const std::size_t j) const
    {
        return {xmin + (double(i) + 0.5)*(xmax-xmin) / nx, ymin + (double(j) + 0.5)*(ymax-ymin) / ny};
    }
    
    self_image_type image()
    {
        self_image_type output;
        output.nx   = nx;
        output.ny   = ny;
        output.xmin = xmin;
        output.xmax = xmax;
        output.ymin = ymin;
        output.ymax = ymax;
        noconst_image_type buf{&raw[0], raw.size()};
        output.raw  = buf;
        return output;
    }
    
    self_const_image_type image() const
    {
        self_const_image_type output;
        output.nx   = nx;
        output.ny   = ny;
        output.xmin = xmin;
        output.xmax = xmax;
        output.ymin = ymin;
        output.ymax = ymax;
        const_image_type buf{&raw[0], raw.size()};
        output.raw  = buf;
        return output;
    }
};