#ifndef FOURVECTOR_H
#define FOURVECTOR_H

#include <iostream>
struct FourVector
{

    int ix, iy, iz, sublattice;

    FourVector()
        : ix(0), iy(0), iz(0), sublattice(0)
    {
    }
    FourVector(int ix, int iy, int iz, int sublattice)
        : ix(ix), iy(iy), iz(iz), sublattice(sublattice)
    {
    }

    FourVector(const FourVector &fv)
        : ix(fv.ix), iy(fv.iy), iz(fv.iz), sublattice(fv.sublattice)
    {
#ifdef PRINT_DEBUG_EXTRA
        std::cout << "Copied 4-vector!" << std::endl;
#endif
    }
};

#endif
