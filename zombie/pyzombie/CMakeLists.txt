cmake_minimum_required(VERSION 3.1.0)
project(pyzombie)

option(PYZOMBIE_BUILD_LIBRARY "Build python library" ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/pybind11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")

# build demo
if (PYZOMBIE_BUILD_LIBRARY)
	pybind11_add_module(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/api.cpp)
	target_include_directories(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_SOURCE_DIR}/include/)
	target_link_libraries(${PROJECT_NAME} PUBLIC zombie pybind11::headers fcpw)
endif()