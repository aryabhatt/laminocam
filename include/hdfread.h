/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals from the
 * U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 * IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 * the U.S. Government has been granted for itself and others acting on its
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 * to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */
#ifndef TOMOCAM_HDFREAD__H
#define TOMOCAM_HDFREAD__H

#include <stdexcept>
#include <string>
#include <vector>

#include <hdf5.h>

#include "array.h"

namespace tomocam::h5 {

    // Read /images dataset (nangles x nrows x ncols), converting to float.
    inline Array<float> read_images(const std::string &filename,
                                    const std::string &dataset = "/images") {
        hid_t fp = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (fp < 0)
            throw std::runtime_error("cannot open HDF5 file: " + filename);

        hid_t ds = H5Dopen2(fp, dataset.c_str(), H5P_DEFAULT);
        if (ds < 0) {
            H5Fclose(fp);
            throw std::runtime_error("cannot open dataset: " + dataset);
        }

        hid_t sp = H5Dget_space(ds);
        hsize_t dims[3] = {0, 0, 0};
        int ndims = H5Sget_simple_extent_dims(sp, dims, nullptr);
        H5Sclose(sp);

        if (ndims != 3) {
            H5Dclose(ds);
            H5Fclose(fp);
            throw std::runtime_error("expected 3-D dataset: " + dataset);
        }

        Array<float> out(dims[0], dims[1], dims[2]);
        H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.begin());
        H5Dclose(ds);
        H5Fclose(fp);
        return out;
    }

    // Read /coords/alpha (or any 1-D dataset) as a vector of floats.
    inline std::vector<float> read_angles(const std::string &filename,
                                          const std::string &dataset = "/coords/alpha") {
        hid_t fp = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (fp < 0)
            throw std::runtime_error("cannot open HDF5 file: " + filename);

        hid_t ds = H5Dopen2(fp, dataset.c_str(), H5P_DEFAULT);
        if (ds < 0) {
            H5Fclose(fp);
            throw std::runtime_error("cannot open dataset: " + dataset);
        }

        hid_t sp = H5Dget_space(ds);
        hsize_t dims[1] = {0};
        int ndims = H5Sget_simple_extent_dims(sp, dims, nullptr);
        H5Sclose(sp);

        if (ndims != 1) {
            H5Dclose(ds);
            H5Fclose(fp);
            throw std::runtime_error("expected 1-D dataset: " + dataset);
        }

        std::vector<float> out(dims[0]);
        H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
        H5Dclose(ds);
        H5Fclose(fp);
        return out;
    }

} // namespace tomocam::h5
#endif // TOMOCAM_HDFREAD__H
