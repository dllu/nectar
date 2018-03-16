cmake_minimum_required(VERSION 2.8)
project(nectar)

find_package(Eigen3 REQUIRED)
find_package(PNG REQUIRED)

set(NECTAR_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third-party/tinypng
    ${EIGEN3_INCLUDE_DIR}
    ${PNG_INCLUDE_DIRS}
    /usr/local/include/libalkusb3-1.0
)

set(NECTAR_LIBRARIES
    ${EIGEN3_LIBRARIES}
    ${PNG_LIBRARIES}
    /usr/local/lib/libalkusb3.so
)

################################################################

set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -Og -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math")

################################################################

include_directories(${NECTAR_INCLUDE_DIRS})

#add_executable(necta
#src/main.cpp
#)
#target_link_libraries(necta ${NECTAR_LIBRARIES})
#add_executable(capture
#src/capture.cpp
#)
#target_link_libraries(capture ${NECTAR_LIBRARIES})
add_executable(demosaic
    src/demosaic.cpp
    third-party/tinypng/pixel.cc
    third-party/tinypng/tinypng.cc
    )
target_link_libraries(demosaic ${NECTAR_LIBRARIES})
