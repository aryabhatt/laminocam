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

#include "gpu/padding.h"
#include <stdexcept>

namespace tomocam::gpu {

    template <typename T>
    __global__ void pad2d_kernel(DevicePtr<const T> input, DevicePtr<T> output,
                                 int3 offset) {
        int3 idx = Index3D();
        if (idx < input.dims()) {
            int3 out_idx = {idx.x + offset.x, idx.y + offset.y, idx.z + offset.z};
            output[out_idx] = input[idx];
        }
    }

    /**
        * Pad intput detector images with zeros in prevent aliasing.
        * @param input The input 3D array to be padded.
        * @param pad_factor The factor by which to pad the array. For example, a
        * pad_factor of 2 will double the size of the array in each dimension.
        * @param type The type of padding to apply (symmetric, left, or right).
        * @return A new DeviceArray containing the padded data.
        */
    template <typename T>
    DeviceArray<T> pad2d(const DeviceArray<T> &input, float pad_factor, PadType type) {

        dims_t dims = input.dims();
        size_t pad_n2 = 2 * static_cast<size_t>(dims.n2 * (pad_factor - 1) / 2);
        size_t pad_n3 = 2 * static_cast<size_t>(dims.n3 * (pad_factor - 1) / 2);
        dims_t out_dims = {dims.n1, dims.n2 + pad_n2, dims.n3 + pad_n3};
        int3 offset = {0, 0, 0};
        if (type == PadType::SYMMETRIC) {
            offset.y = pad_n2 / 2;
            offset.z = pad_n3 / 2;
        } else if (type == PadType::LEFT) {
            offset.y = pad_n2;
            offset.z = pad_n3;
        } else if (type == PadType::RIGHT) {
            // do nothing, offset is already 0
        }
        DeviceArray<T> output(out_dims);
        dim3 blockSize = dim3(1, 8, 32);
        dim3 gridSize = make_grid(dims, blockSize);
        pad2d_kernel<T><<<gridSize, blockSize>>>(input, output, offset);
        SAFE_CALL(cudaGetLastError());
        return output;
    }

    template DeviceArray<float> pad2d(const DeviceArray<float> &input,
                                      float pad_factor, PadType type);
    template DeviceArray<double> pad2d(const DeviceArray<double> &input,
                                       float pad_factor, PadType type);

    template <typename T>
    __global__ void crop3d_kernel(DevicePtr<const T> input, DevicePtr<T> output,
                                  int3 offset) {
        int3 idx = Index3D();
        if (idx < output.dims()) {
            int3 in_idx = {idx.x + offset.x, idx.y + offset.y, idx.z + offset.z};
            output[idx] = input[in_idx];
        }
    }

    template <typename T>
    DeviceArray<T> crop3d(const DeviceArray<T> &input, dims_t out_dims,
                          PadType type) {
        dims_t dims = input.dims();
        if (dims.n1 < out_dims.n1 || dims.n2 < out_dims.n2 ||
            dims.n3 < out_dims.n3) {
            throw std::invalid_argument(
                "Crop dimensions must be smaller than input dimensions.");
        }

        int3 offset = {(int)(dims.n1 - out_dims.n1), (int)(dims.n2 - out_dims.n2),
                       (int)(dims.n3 - out_dims.n3)};
        if (type == PadType::SYMMETRIC) {
            offset.x /= 2;
            offset.y /= 2;
            offset.z /= 2;
        } else if (type == PadType::LEFT) {
            offset.x = 0;
            offset.y = 0;
            offset.z = 0;
        }
        DeviceArray<T> output(out_dims);
        dim3 blockSize = dim3(1, 8, 32);
        dim3 gridSize = make_grid(out_dims, blockSize);
        crop3d_kernel<T><<<gridSize, blockSize>>>(input, output, offset);
        SAFE_CALL(cudaGetLastError());
        return output;
    }

    template DeviceArray<float> crop3d(const DeviceArray<float> &input,
                                       dims_t out_dims, PadType type);
    template DeviceArray<double> crop3d(const DeviceArray<double> &input,
                                        dims_t out_dims, PadType type);

    template <typename T>
    DeviceArray<T> crop3d(const DeviceArray<T> &input, dims_t out_dims,
                          dims_t offset) {
        int3 off = {(int)offset.n1, (int)offset.n2, (int)offset.n3};
        DeviceArray<T> output(out_dims);
        dim3 blockSize = dim3(1, 8, 32);
        dim3 gridSize = make_grid(out_dims, blockSize);
        crop3d_kernel<T><<<gridSize, blockSize>>>(input, output, off);
        SAFE_CALL(cudaGetLastError());
        return output;
    }

    template DeviceArray<float> crop3d(const DeviceArray<float> &input,
                                       dims_t out_dims, dims_t offset);
    template DeviceArray<double> crop3d(const DeviceArray<double> &input,
                                        dims_t out_dims, dims_t offset);

    template <typename T>
    DeviceArray<T> pad3d(const DeviceArray<T> &input, dims_t out_dims, PadType type) {
        dims_t dims = input.dims();
        int3 offset = {0, 0, 0};
        if (type == PadType::SYMMETRIC) {
            offset.x = static_cast<int>((out_dims.n1 - dims.n1) / 2);
            offset.y = static_cast<int>((out_dims.n2 - dims.n2) / 2);
            offset.z = static_cast<int>((out_dims.n3 - dims.n3) / 2);
        } else if (type == PadType::LEFT) {
            offset.x = static_cast<int>(out_dims.n1 - dims.n1);
            offset.y = static_cast<int>(out_dims.n2 - dims.n2);
            offset.z = static_cast<int>(out_dims.n3 - dims.n3);
        }
        DeviceArray<T> output(out_dims);
        dim3 blockSize = dim3(1, 8, 32);
        dim3 gridSize = make_grid(dims, blockSize);
        pad2d_kernel<T><<<gridSize, blockSize>>>(input, output, offset);
        SAFE_CALL(cudaGetLastError());
        return output;
    }

    template DeviceArray<float> pad3d(const DeviceArray<float> &, dims_t, PadType);
    template DeviceArray<double> pad3d(const DeviceArray<double> &, dims_t, PadType);
} // namespace tomocam::gpu
