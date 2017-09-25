cmake_minimum_required(VERSION 2.8)
project(necta)

find_package(OpenCV REQUIRED)
set(NECTA_INCLUDE_DIRS
	${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/include/libalkusb3-1.0
)
set(NECTA_LIBRARIES
    ${OpenCV_LIBS}
    /usr/local/lib/libalkusb3.so
)

################################################################

set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -Og -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

################################################################

include_directories(${NECTA_INCLUDE_DIRS})
link_directories(${NECTA_LIBRARY_DIRS})

add_executable(necta
    src/main.cpp
)
target_link_libraries(necta ${NECTA_LIBRARIES})
add_executable(capture
    src/capture.cpp
)
target_link_libraries(capture ${NECTA_LIBRARIES})

