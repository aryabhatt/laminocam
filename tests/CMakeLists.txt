
add_executable(test_cg_normal test_cg_normal.cpp)
target_link_libraries(test_cg_normal PUBLIC tomocam_scalar)
add_test(NAME cg_normal COMMAND test_cg_normal)

add_executable(test_adjoint test_adjoint.cpp)
target_link_libraries(test_adjoint PUBLIC tomocam_scalar)
add_test(NAME adjoint COMMAND test_adjoint)

add_executable(test_sysmat_toeplitz test_sysmat_toeplitz.cpp)
target_link_libraries(test_sysmat_toeplitz PUBLIC tomocam_scalar)
add_test(NAME sysmat_toeplitz COMMAND test_sysmat_toeplitz)

if (USE_GPU)
    add_executable(test_proj_gpu_cpu test_proj_gpu_cpu.cu)
    target_link_libraries(test_proj_gpu_cpu PUBLIC tomocam_scalar)
    add_test(NAME proj_gpu_cpu COMMAND test_proj_gpu_cpu)

    add_executable(test_nufft_gpu test_nufft_gpu.cu)
    target_link_libraries(test_nufft_gpu PUBLIC tomocam_scalar)
    add_test(NAME nufft_gpu COMMAND test_nufft_gpu)

    add_executable(test_cg_gpu_cpu test_cg_gpu_cpu.cu)
    target_link_libraries(test_cg_gpu_cpu PUBLIC tomocam_scalar)
    add_test(NAME cg_gpu_cpu COMMAND test_cg_gpu_cpu)

    add_executable(test_bregman_gpu_cpu test_bregman_gpu_cpu.cu)
    target_link_libraries(test_bregman_gpu_cpu PUBLIC tomocam_scalar)
    add_test(NAME bregman_gpu_cpu COMMAND test_bregman_gpu_cpu)

    add_executable(test_cg test_cg.cu)
    target_link_libraries(test_cg PUBLIC tomocam_scalar)
    add_test(NAME cg COMMAND test_cg)

    add_executable(test_sysmat_toeplitz_gpu test_sysmat_toeplitz_gpu.cu)
    target_link_libraries(test_sysmat_toeplitz_gpu PUBLIC tomocam_scalar)
    add_test(NAME sysmat_toeplitz_gpu COMMAND test_sysmat_toeplitz_gpu)

    add_executable(test_device_array_ops test_device_array_ops.cu)
    target_link_libraries(test_device_array_ops PUBLIC tomocam_scalar)
    add_test(NAME device_array_ops COMMAND test_device_array_ops)
endif()
