# *****************************************************************************
# Copyright 2018-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# *****************************************************************************

# SEAL requires >= 3.10
cmake_minimum_required(VERSION 3.10)

# Global project name
project(he_transformer LANGUAGES CXX)

# Cmake flags SEAL requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INSTALL_MESSAGE LAZY)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -march=native -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")

# This allows libhe_seal_backend.so to find libraries in the same directory
set(CMAKE_INSTALL_RPATH "\$ORIGIN")

# These variables are undocumented but useful.
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
  message(STATUS "Setting clang flags...")
  include(cmake/clang_flags.cmake)
endif()

if(CMAKE_BUILD_TYPE)
  set(RELEASE_TYPES
      Debug
      Release
      RelWithDebInfo
      MinSizeRel)
  list(FIND RELEASE_TYPES ${CMAKE_BUILD_TYPE} INDEX_FOUND)
  if(${INDEX_FOUND} EQUAL -1)
    message(
      FATAL_ERROR
        "CMAKE_BUILD_TYPE must be one of Debug, Release, RelWithDebInfo, or MinSizeRel"
      )
  endif()
endif()

include(cmake/sdl.cmake)

# For ngraph/log
set(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DPROJECT_ROOT_DIR="${PROJECT_ROOT_DIR}")

# he-transformer headers
set(HE_TRANSFORMER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# External projects (ngraph, seal, etc. )
set(EXTERNAL_INSTALL_DIR ${CMAKE_BINARY_DIR}/external)
set(EXTERNAL_INSTALL_LIB_DIR ${EXTERNAL_INSTALL_DIR}/lib)
set(EXTERNAL_INSTALL_INCLUDE_DIR ${EXTERNAL_INSTALL_DIR}/include)

set(NGRAPH_HE_INSTALL_DOC ${CMAKE_BINARY_DIR}/doc)

if(NOT DEFINED EXTERNAL_PROJECTS_ROOT)
  set(EXTERNAL_PROJECTS_ROOT ${CMAKE_CURRENT_BINARY_DIR})
endif()

# Build options python version used in virtual environment
find_package(PythonInterp 3)
if(PYTHONINTERP_FOUND)
  message(STATUS "PYTHON_VERSION_STRING " ${PYTHON_VERSION_STRING})
elseif()
  message(FATAL_ERROR "Python3 not found.")
endif()

if(NOT DEFINED PYTHON_VENV_VERSION)
  set(PYTHON_VENV_VERSION
      "python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
endif()
message(STATUS "PYTHON_VENV_VERSION ${PYTHON_VENV_VERSION}")

option(NGRAPH_HE_SANITIZE_ADDRESS "Enable address sanitizer" OFF)
option(NGRAPH_HE_DOC_BUILD_ENABLE "Enable building of documentation" OFF)

message(STATUS "NGRAPH_HE_DOC_BUILD_ENABLE: ${NGRAPH_HE_DOC_BUILD_ENABLE}")

if(NGRAPH_HE_SANITIZE_ADDRESS)
  message(STATUS "NGRAPH_HE_SANITIZE_ADDRESS")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
endif()

# Get OS version
if(NOT APPLE)
  execute_process(COMMAND cat /etc/os-release
                  OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "\""
                 ""
                 LSB_RELEASE_ID_SHORT
                 ${LSB_RELEASE_ID_SHORT})
  string(REGEX MATCH
               "ID=\([a-z])+"
               OS_VERSION
               "${LSB_RELEASE_ID_SHORT}")
  string(REGEX MATCH
               "([a-z])+"
               OS_VERSION
               "${OS_VERSION}")
  message(STATUS "OS version: ${OS_VERSION}")
else()
  # Handle the case for MacOS TBD
endif()

include(cmake/ngraph-tf.cmake)
include(cmake/gtest.cmake)
include(cmake/json.cmake)
include(cmake/openmp.cmake)
include(cmake/seal.cmake)
include(cmake/boost.cmake)
include(cmake/protobuf.cmake)

# HE transformer source and test directories
add_subdirectory(src)
add_subdirectory(test)

if(NGRAPH_HE_DOC_BUILD_ENABLE)
  add_subdirectory(doc)
endif()

# For python bindings
add_subdirectory(python)
