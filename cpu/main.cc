#include "grid.hpp"
#include "complex.hpp"
#include "vtkout.hpp"

int main(int argc, char** argv)
{
    const int n = 3000;
    grid_t arr(0.0, n, n, -0.698, -0.6964, 0.444, 0.4448);
    
    
    const int max_its = 250;
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            complex_t c = arr.get_c(i, j);
            complex_t z{0.0, 0.0};
            int it = 0;
            while ((++it < max_its) && (~z*z).x < 4.0)
            {
                z = z*z + c;
            }
            if (it == max_its) it = -1;
            arr(i, j) = double(it)/max_its;
        }
    }
    
    vtkout("out.vtk", arr);
    
    return 0;
}