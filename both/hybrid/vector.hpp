#pragma once

#include "cuda_stuff.h"
#include "hybrid/device.hpp"

namespace hybrid
{
    template <typename data_t>
    struct device_allocator_t
    {
        using value_type = data_t;
        device_allocator_t() = default;
        template<typename rhs_t>
        constexpr device_allocator_t (const device_allocator_t <rhs_t>&) noexcept {}
        [[nodiscard]] data_t* allocate(std::size_t n)
        {
            data_t* output = nullptr;
            if (n==0) return output;
            std::string err_string = "attempted to allocate device memory without device support";
#if(_gt_cuda)
            auto er_code = cudaMalloc(&output, n*sizeof(data_t));
#endif
            return output;
        }
        
        void deallocate(data_t* p, std::size_t n) noexcept
        {
#if(_gt_cuda)
            auto er_code = cudaFree(p);
            //Done to avoid annoying compiler warning.
            p = nullptr;
#endif
        }
    };
    
    template <typename data_t>
    struct device_vector
    {
        using value_type = data_t;
        using allocator_t = device_allocator_t<data_t>;
        allocator_t allocator;
        data_t* raw = nullptr;
        std::size_t c_size = 0;
        
        device_vector(){}
        device_vector(const std::size_t n) 
        {
            this->resize(n);
        }
        
        device_vector& operator = (const device_vector& rhs)
        {
            if (rhs.size() == 0) return *this;
            this->resize(rhs.size());
#if (_gt_cuda)
            if (rhs.size() > 0) cudaMemcpy(this->raw, rhs.raw, c_size*sizeof(value_type), cudaMemcpyDeviceToDevice);
#endif
            return *this;
        }
        
        std::size_t size() const {return c_size;}
        
        void resize(const std::size_t& n)
        {
#if (_gt_cuda)
            if (raw != nullptr) allocator.deallocate(raw, c_size);
            c_size = n;
            raw = allocator.allocate(n);
#endif
        }
        
        ~device_vector()
        {
            if (raw != nullptr) allocator.deallocate(raw, c_size);
        }
        
        operator std::vector<data_t>() const
        {
            std::vector<data_t> output;
            output.resize(this->size());
#if (_gt_cuda)
            cudaMemcpy(&output[0], raw, c_size*sizeof(value_type), cudaMemcpyDeviceToHost);
#endif
            return output;
        }
        
        _gt_hybrid data_t& operator[]       (const std::size_t& idx)       { return raw[idx]; }
        _gt_hybrid const data_t& operator[] (const std::size_t& idx) const { return raw[idx]; }
    };
    
    template <typename data_t, typename device_t> using vector_t
        = typename std::conditional<is_gpu<device_t>, device_vector<data_t>, std::vector<data_t>>::type;
    
    template <typename data_t>
    struct vector_image_t
    {
        data_t* raw;
        std::size_t c_size;
        std::size_t size() const { return c_size; }
        _gt_hybrid data_t& operator[] (const std::size_t& idx) { return raw[idx]; }
    };
    
    template <typename data_t>
    struct const_vector_image_t
    {
        const data_t* raw;
        std::size_t c_size;
        std::size_t size() const { return c_size; }
        _gt_hybrid const data_t& operator[] (const std::size_t& idx) const { return raw[idx]; }
    };
}