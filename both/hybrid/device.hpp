#pragma once

#include <type_traits>
#include "cuda_stuff.h"

namespace hybrid
{
    static constexpr struct cpu_t {} cpu;
    static constexpr struct gpu_t {} gpu;
    
    using best_t = typename std::conditional<_gt_cuda, gpu_t, cpu_t>::type;
    static best_t best;
    
    template <typename T> constexpr static bool is_gpu = std::is_same<T, gpu_t>::value;
}