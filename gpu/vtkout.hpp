#pragma once

#include <string>
#include <fstream>
#include "grid.hpp"

template <typename data_t>
static void vtkout(const std::string& fname, const grid_t<data_t>& arr)
{
    std::ofstream mf(fname);
    mf << "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS " << (arr.nx+1) << " " << (arr.ny+1) << " 1\n";
    mf << "ORIGIN " << arr.xmin << " " << arr.ymin << " 0.0\n";
    mf << "SPACING " << (arr.xmax-arr.xmin)/arr.nx << " " << (arr.ymax-arr.ymin)/arr.ny << " 1.0\n";
    mf << "CELL_DATA " << arr.nx*arr.ny << "\nSCALARS data double\nLOOKUP_TABLE default\n";
    for (const auto& d: arr.raw) mf << d << "\n";
}