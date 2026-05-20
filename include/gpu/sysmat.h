
#ifndef TOMOCAM_GPU_SYSMAT_H
#define TOMOCAM_GPU_SYSMAT_H

#include "gpu/device_array.h"
#include "gpu/nufft.h"
#include "gpu/polar_grid.h"

namespace tomocam::gpu {

    template <typename T>
    using complex = cuda::std::complex<T>;

    template <typename T>
    DeviceArray<T> sysmat(const DeviceArray<T> &m, const PolarGrid<T> &pgrid) {

        // cast m to complex type
        auto m_cplx = array::to_complex(m);
        // allocate complex arrays for projections
        DeviceArray<complex<T>> p_cplx(pgrid.dims());

        //  call type-2 NUFFT
        nufft::nufft3d2(p_cplx, m_cplx, pgrid);

        // call type-1 NUFFT (Adjoint)
        DeviceArray<complex<T>> a_cplx(m.dims());
        nufft::nufft3d1(p_cplx, a_cplx, pgrid);

        // scale and cast to real type (matches CPU sysmat: scale = ntheta * ncols)
        auto result = array::to_real(a_cplx);
        T scale = static_cast<T>(pgrid.x.nslices()) * static_cast<T>(pgrid.x.ncols());
        array::scale_inplace(result, T(1) / scale);
        return result;
    }
} // namespace tomocam::gpu
#endif // TOMOCAM_GPU_SYSMAT_H
