

add_executable(backproj backproj.cpp)
target_link_libraries(backproj PRIVATE tomlplusplus::tomlplusplus)
target_link_libraries(backproj PUBLIC tomocam_scalar)
target_link_libraries(backproj PUBLIC TIFF::TIFF)

add_executable(test_gradient test_grad.cpp)
target_link_libraries(test_gradient PUBLIC tomocam_scalar)

add_executable(test_loss test_loss.cpp)
target_link_libraries(test_loss PUBLIC tomocam_scalar)

add_executable(test_lipschitz test_lipschitz.cpp)
target_link_libraries(test_lipschitz PUBLIC tomocam_scalar)

add_executable(test_nufft test_nufft.cpp)
target_link_libraries(test_nufft PUBLIC tomocam_scalar)

