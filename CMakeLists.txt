﻿# CMakeList.txt: CWENO_2D 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

project ("CWENO_2D")
#添加python路径，为了使用matplotlibcpp
include_directories(D:/user/Python/include)
include_directories(D:\\user\\Python/Lib/site-packages/numpy/core/include)
link_directories(D:\\user\\Python/libs)

#添加头文件定义
add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)
# 将源代码添加到此项目的可执行文件。
add_executable (CWENO_2D "CWENO_2D.cpp" "CWENO_2D.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CWENO_2D PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。
