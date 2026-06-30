

add_executable(backproj backproj.cpp)
target_link_libraries(backproj PRIVATE tomlplusplus::tomlplusplus)
target_link_libraries(backproj PUBLIC tomocam_scalar)
target_link_libraries(backproj PUBLIC TIFF::TIFF)

add_executable(test_nufft test_nufft.cpp)
target_link_libraries(test_nufft PUBLIC tomocam_scalar)

add_executable(test_adjoint test_adjoint.cpp)
target_link_libraries(test_adjoint PUBLIC tomocam_scalar)

add_executable(test_sysmat_toeplitz test_sysmat_toeplitz.cpp)
target_link_libraries(test_sysmat_toeplitz PUBLIC tomocam_scalar)

add_executable(test_gradient test_grad.cpp)
target_link_libraries(test_gradient PUBLIC tomocam_scalar)

if (USE_GPU)
    add_executable(test_proj_gpu_cpu test_proj_gpu_cpu.cu)
    target_link_libraries(test_proj_gpu_cpu PUBLIC tomocam_scalar)

    add_executable(test_loss test_loss.cpp)
    target_link_libraries(test_loss PUBLIC tomocam_scalar)

    add_executable(test_nufft_gpu test_nufft_gpu.cu)
    target_link_libraries(test_nufft_gpu PUBLIC tomocam_scalar)

    add_executable(test_cg_gpu_cpu test_cg_gpu_cpu.cu)
    target_link_libraries(test_cg_gpu_cpu PUBLIC tomocam_scalar)

    add_executable(test_bregman_gpu_cpu test_bregman_gpu_cpu.cu)
    target_link_libraries(test_bregman_gpu_cpu PUBLIC tomocam_scalar)

    add_executable(test_cg test_cg.cu)
    target_link_libraries(test_cg PUBLIC tomocam_scalar)
endif()

