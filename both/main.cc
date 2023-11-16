#include <chrono>
#include "grid.hpp"
#include "complex.hpp"
#include "vtkout.hpp"
#include "hybrid/device.hpp"
#include "hybrid/range.hpp"
#include "hybrid/execute.hpp"

struct tmr_t
{
    std::string name;
    decltype(std::chrono::high_resolution_clock::now()) tstart, tend;
    tmr_t(const std::string& n_in) : name{n_in} {}
    void start()
    {
        tstart = std::chrono::high_resolution_clock::now();
    }
    
    void stop()
    {
        tend = std::chrono::high_resolution_clock::now();
    }
};

std::ostream& operator << (std::ostream& os, const tmr_t& tmr)
{
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tmr.tend - tmr.tstart);
    os << tmr.name << " took " <<  duration.count() << " ms";
    return os;
}

int main(int argc, char** argv)
{
    const int n = 3000;
    auto device = hybrid::best;
    grid_t arr(0.0, n, n, -0.698, -0.6964, 0.444, 0.4448, device);
    
    hybrid::range_t range{arr.nx, arr.ny};
    const int max_its = 2500;
    auto arr_img = arr.image();
    
    auto loop = [=] _gt_hybrid (const int i[2]) mutable
    {
        complex_t c = arr_img.get_c(i[0], i[1]);
        complex_t z{0.0, 0.0};
        int it = 0;
        while ((++it < max_its) && (~z*z).x < 4.0)
        {
            z = z*z + c;
        }
        if (it == max_its) it = -1;
        arr_img(i[0], i[1]) = double(it)/max_its;
    };
    
    tmr_t tmr("kernel");
    tmr.start();
    hybrid::execute(range, loop, device);
    tmr.stop();
    std::cout << tmr << std::endl;
    
    vtkout("out.vtk", arr);
    
    return 0;
}