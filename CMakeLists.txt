cmake_minimum_required(VERSION 3.16)

project(eval_core
    VERSION 0.1.0
    DESCRIPTION "Batched LLM inference for AG News classification"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Put our .so next to our own build dir, not the top-level one.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# ---------------------------------------------------------------------------
# Locate llama.cpp
#   * Preferred: pass -DLLAMA_PATH=/abs/path/to/llama.cpp
#   * Fallback : the LLAMA_PATH environment variable
#   * Default  : ../third_party/llama.cpp relative to this file
# ---------------------------------------------------------------------------
if(NOT DEFINED LLAMA_PATH OR LLAMA_PATH STREQUAL "")
    if(DEFINED ENV{LLAMA_PATH} AND NOT "$ENV{LLAMA_PATH}" STREQUAL "")
        set(LLAMA_PATH "$ENV{LLAMA_PATH}")
        message(STATUS "eval_core: using LLAMA_PATH from environment: ${LLAMA_PATH}")
    else()
        set(LLAMA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp")
        message(STATUS "eval_core: using default LLAMA_PATH: ${LLAMA_PATH}")
    endif()
endif()

get_filename_component(LLAMA_PATH "${LLAMA_PATH}" ABSOLUTE)

if(NOT IS_DIRECTORY "${LLAMA_PATH}")
    message(FATAL_ERROR
        "LLAMA_PATH is not a directory: '${LLAMA_PATH}'\n"
        "Point it at the root of your llama.cpp clone "
        "(the folder that contains include/llama.h).")
endif()

if(NOT EXISTS "${LLAMA_PATH}/CMakeLists.txt")
    message(FATAL_ERROR
        "'${LLAMA_PATH}' does not look like a llama.cpp checkout: "
        "CMakeLists.txt is missing.\n"
        "Did you point LLAMA_PATH at a subdirectory (src, build, include)? "
        "Use the repository root instead.")
endif()
if(NOT EXISTS "${LLAMA_PATH}/include/llama.h")
    message(FATAL_ERROR
        "'${LLAMA_PATH}' is missing include/llama.h.\n"
        "This can happen with a partial or corrupted clone. "
        "Try: git -C '${LLAMA_PATH}' status")
endif()

message(STATUS "eval_core: LLAMA_PATH = ${LLAMA_PATH}")

# ---------------------------------------------------------------------------
# Configure llama.cpp before add_subdirectory
# ---------------------------------------------------------------------------
set(BUILD_SHARED_LIBS     ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_PATH}" "${CMAKE_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Detect the KV-cache clear API exposed by this llama.cpp revision.
# ---------------------------------------------------------------------------
set(_llama_h "${LLAMA_PATH}/include/llama.h")
file(READ "${_llama_h}" _llama_h_contents)

if(_llama_h_contents MATCHES "llama_memory_clear\\s*\\(")
    set(EVALCORE_KV_API "NEW")
    message(STATUS "eval_core: using llama_memory_clear (new memory API)")
elseif(_llama_h_contents MATCHES "llama_kv_self_clear\\s*\\(")
    set(EVALCORE_KV_API "INTERMEDIATE")
    message(STATUS "eval_core: using llama_kv_self_clear")
elseif(_llama_h_contents MATCHES "llama_kv_cache_clear\\s*\\(")
    set(EVALCORE_KV_API "OLD")
    message(STATUS "eval_core: using llama_kv_cache_clear (legacy)")
else()
    message(FATAL_ERROR
        "Could not find any KV-cache clear function in ${_llama_h}.\n"
        "Please check the llama.cpp revision you're building against.")
endif()

# ---------------------------------------------------------------------------
# Our library
# ---------------------------------------------------------------------------
add_library(eval_core SHARED
    cpp/eval_core.cpp
    cpp/eval_core.hpp
)
add_library(eval_core::eval_core ALIAS eval_core)

target_include_directories(eval_core PRIVATE
    "${LLAMA_PATH}/include"
    "${LLAMA_PATH}/ggml/include"
)

target_include_directories(eval_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(eval_core PRIVATE llama)

if(EVALCORE_KV_API STREQUAL "NEW")
    target_compile_definitions(eval_core PRIVATE EVALCORE_LLAMA_NEW_MEMORY_API)
elseif(EVALCORE_KV_API STREQUAL "INTERMEDIATE")
    target_compile_definitions(eval_core PRIVATE EVALCORE_LLAMA_KV_SELF_API)
endif()

target_compile_options(eval_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

# No BUILD_RPATH override. CMake auto-adds the correct build-tree RPATH
# when `llama` is a target in the same build. Setting it manually here
# pointed at ${LLAMA_PATH}/build/src, which does not exist under
# add_subdirectory() and shadowed the correct auto-generated entries.

set_target_properties(eval_core PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS eval_core
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES eval_core.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "eval_core configuration summary")
message(STATUS "  Build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard     : ${CMAKE_CXX_STANDARD}")
message(STATUS "  llama.cpp path   : ${LLAMA_PATH}")
message(STATUS "  KV-cache API     : ${EVALCORE_KV_API}")
message(STATUS "  Output library   : ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}eval_core${CMAKE_SHARED_LIBRARY_SUFFIX}")
message(STATUS "")