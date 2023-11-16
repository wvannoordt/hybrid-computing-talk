#include "grid.hpp"
#include "complex.hpp"
#include "vtkout.hpp"

template <typename data_t>
__global__ void k_mandel(data_t* data, const int nx, const int ny, const double xmin, const double xmax, const double ymin, const double ymax, const int max_its)
{
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    const int j = threadIdx.y + blockIdx.y*blockDim.y;
    complex_t c{xmin + (double(i) + 0.5)*(xmax-xmin) / nx, ymin + (double(j) + 0.5)*(ymax-ymin) / ny};
    
    if ((i < nx) && (j < ny))
    {
        complex_t z{0.0, 0.0};
        int it = 0;
        while ((++it < max_its) && (~z*z).x < 4.0)
        {
            z = z*z + c;
        }
        if (it == max_its) it = -1;
        data[i + nx*j] = double(it)/max_its;
    }
}

int main(int argc, char** argv)
{
    const int n = 3000;
    grid_t arr(0.0, n, n, -0.698, -0.6964, 0.444, 0.4448);

    const int max_its = 250;
    
    const int blk_wid = 8;
    
    dim3 grd_size, blk_size;
    grd_size.x = 1+(arr.nx-1)/blk_wid;
    grd_size.y = 1+(arr.ny-1)/blk_wid;
    grd_size.z = 1;
    blk_size.x = blk_wid;
    blk_size.y = blk_wid;
    blk_size.z = 1;
    k_mandel<<<grd_size, blk_size>>>(arr.d_raw, arr.nx, arr.ny, arr.xmin, arr.xmax, arr.ymin, arr.ymax, max_its);
    cudaMemcpy(&(arr.raw[0]), arr.d_raw, arr.raw.size()*sizeof(double), cudaMemcpyDeviceToHost);
    
    vtkout("out.vtk", arr);
    
    return 0;
}