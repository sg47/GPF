cmake_minimum_required(VERSION 2.8)
project( tracker )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
find_package( OpenCV REQUIRED )
include_directories( /usr/local/include/eigen3 )
add_executable( tracker utils.h tracker.cpp )
target_link_libraries( tracker ${OpenCV_LIBS} )

