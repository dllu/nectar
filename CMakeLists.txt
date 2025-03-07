cmake_minimum_required(VERSION 3.11)
project(necta)
set(OpenGL_GL_PREFERENCE GLVND)

find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
set(NECTA_INCLUDE_DIRS
	${PROJECT_SOURCE_DIR}/src
    ${OPENGL_INCLUDE_DIR}
    ${SDL2_INCLUDE_DIRS}
    /usr/include/libalkusb3/
)
set(NECTA_LIBRARIES
    ${OPENGL_LIBRARIES}
    ${SDL2_LIBRARIES}
    /usr/lib/libalkusb3.so.1.11.0
)

################################################################

set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -Og -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

################################################################

include_directories(${NECTA_INCLUDE_DIRS})
link_directories(${NECTA_LIBRARY_DIRS})

include_directories(imgui)
add_executable(necta
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_impl_opengl3.cpp
    imgui/imgui_impl_sdl2.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    src/main.cpp
)
target_link_libraries(necta ${NECTA_LIBRARIES})
