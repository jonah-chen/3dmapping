#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(RealSenseKinfuExample)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(OpenGL_GL_PREFERENCE GLVND)

find_package(realsense2 REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)

set(DEPENDENCIES glfw pthread ${OPENGL_LIBRARIES} ${OpenCV_LIBS} ${realsense2_LIBRARY} ${DEPENDENCIES})

add_executable(rs-kinfu rs-kinfu.cpp)
set_property(TARGET rs-kinfu PROPERTY CXX_STANDARD 11)
target_link_libraries(rs-kinfu ${DEPENDENCIES})
include_directories(helpers ${OpenCV_INCLUDE_DIRS})
set_target_properties (rs-kinfu PROPERTIES
	FOLDER "Examples/OpenCV"
)

install(TARGETS rs-kinfu)
