
# create a simple phantom
add_executable(chessboard chessboard.cpp)
target_link_libraries(chessboard PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(chessboard PUBLIC tomocam)
target_link_libraries(chessboard PUBLIC TIFF::TIFF)

# three-dimnsional shepp-log phantom
add_executable(shepp3d shepplogan.cpp)
target_link_libraries(shepp3d PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(shepp3d PUBLIC tomocam)
target_link_libraries(shepp3d PUBLIC TIFF::TIFF)


add_executable(forward forward.cpp)
target_link_libraries(forward PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(forward PUBLIC tomocam)
target_link_libraries(forward PUBLIC TIFF::TIFF)

add_executable(backproj backproj.cpp)
target_link_libraries(backproj PRIVATE tomlplusplus::tomlplusplus)
target_link_libraries(backproj PUBLIC tomocam)
target_link_libraries(backproj PUBLIC TIFF::TIFF)

add_executable(test_fft test_fft.cpp)
target_link_libraries(test_fft PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_fft PUBLIC tomocam)

add_executable(test_gradient test_grad.cpp)
target_link_libraries(test_gradient PUBLIC tomocam)

add_executable(test_loss test_loss.cpp)
target_link_libraries(test_loss PUBLIC tomocam)

add_executable(test_lipschitz test_lipschitz.cpp)
target_link_libraries(test_lipschitz PUBLIC tomocam)

add_executable(test_nufft test_nufft.cpp)
target_link_libraries(test_nufft PUBLIC tomocam)

add_executable(test_cg test_cg.cpp)
target_link_libraries(test_cg PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_cg PUBLIC tomocam)
target_link_libraries(test_cg PUBLIC TIFF::TIFF)

# add tests in order
add_test(NAME test01 COMMAND chessboard "${CMAKE_SOURCE_DIR}/config/chess01.json")
add_test(NAME test02 COMMAND chessboard "${CMAKE_SOURCE_DIR}/config/chess02.json")
add_test(NAME test03 COMMAND forward    "${CMAKE_SOURCE_DIR}/config/fwd01.json")
add_test(NAME test04 COMMAND forward    "${CMAKE_SOURCE_DIR}/config/fwd02.json")
add_test(NAME test05 COMMAND backproj   "${CMAKE_SOURCE_DIR}/config/bwd01.json")
