#pragma once

#include <string>
#include <fstream>
#include "grid.hpp"

template <typename data_t, typename device_t>
static void vtkout(const std::string& fname, const grid_t<data_t, device_t, false, false>& arr)
{
    std::vector<data_t> raw_data = arr.raw;
    std::ofstream mf(fname);
    mf << "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS " << (arr.nx+1) << " " << (arr.ny+1) << " 1\n";
    mf << "ORIGIN " << arr.xmin << " " << arr.ymin << " 0.0\n";
    mf << "SPACING " << (arr.xmax-arr.xmin)/arr.nx << " " << (arr.ymax-arr.ymin)/arr.ny << " 1.0\n";
    mf << "CELL_DATA " << arr.nx*arr.ny << "\nSCALARS data double\nLOOKUP_TABLE default\n";
    for (const auto& d: raw_data) mf << d << "\n";
}