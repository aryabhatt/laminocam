# CMake generated Testfile for 
# Source directory: /home/dkumar/XMCD/scalar/tests
# Build directory: /home/dkumar/XMCD/scalar/debug/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[test01]=] "/home/dkumar/XMCD/scalar/debug/tests/chessboard" "/home/dkumar/XMCD/scalar/config/chess01.json")
set_tests_properties([=[test01]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/dkumar/XMCD/scalar/tests/CMakeLists.txt;45;add_test;/home/dkumar/XMCD/scalar/tests/CMakeLists.txt;0;")
add_test([=[test02]=] "/home/dkumar/XMCD/scalar/debug/tests/chessboard" "/home/dkumar/XMCD/scalar/config/chess02.json")
set_tests_properties([=[test02]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/dkumar/XMCD/scalar/tests/CMakeLists.txt;46;add_test;/home/dkumar/XMCD/scalar/tests/CMakeLists.txt;0;")
add_test([=[test03]=] "/home/dkumar/XMCD/scalar/debug/tests/forward" "/home/dkumar/XMCD/scalar/config/fwd01.json")
set_tests_properties([=[test03]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/dkumar/XMCD/scalar/tests/CMakeLists.txt;47;add_test;/home/dkumar/XMCD/scalar/tests/CMakeLists.txt;0;")
add_test([=[test04]=] "/home/dkumar/XMCD/scalar/debug/tests/forward" "/home/dkumar/XMCD/scalar/config/fwd02.json")
set_tests_properties([=[test04]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/dkumar/XMCD/scalar/tests/CMakeLists.txt;48;add_test;/home/dkumar/XMCD/scalar/tests/CMakeLists.txt;0;")
add_test([=[test05]=] "/home/dkumar/XMCD/scalar/debug/tests/backproj" "/home/dkumar/XMCD/scalar/config/bwd01.json")
set_tests_properties([=[test05]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/dkumar/XMCD/scalar/tests/CMakeLists.txt;49;add_test;/home/dkumar/XMCD/scalar/tests/CMakeLists.txt;0;")
