cmake_minimum_required(VERSION 3.21)
project(fasteda VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Release build flags ──────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/O2 /W4 /arch:AVX2)
else()
  add_compile_options(-O2 -Wall -Wextra -march=native)
endif()

# ── nanobind (as git submodule) ──────────────────────────────────
add_subdirectory(extern/nanobind)

# ── Source files ────────────────────────────────────────────────
set(FASTEDA_SOURCES
    src/core/stats_engine.cpp
    src/core/stream_reader.cpp
    src/core/profile_builder.cpp
    src/core/report_serializer.cpp
    src/bindings/bindings.cpp
)

# ── nanobind Python extension ────────────────────────────────────
nanobind_add_module(fasteda_core ${FASTEDA_SOURCES})

target_include_directories(fasteda_core PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# ── Standalone benchmark binary (no Python needed) ───────────────
add_executable(fasteda_bench
    src/core/stats_engine.cpp
    src/core/stream_reader.cpp
    src/core/profile_builder.cpp
    benchmarks/bench_main.cpp
)

target_include_directories(fasteda_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(fasteda_bench PRIVATE FASTEDA_STANDALONE)

# ── Tests (GoogleTest) ───────────────────────────────────────────
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(fasteda_tests
    src/core/stats_engine.cpp
    src/core/stream_reader.cpp
    src/core/profile_builder.cpp
    tests/cpp/test_stats_engine.cpp
    tests/cpp/test_hyperloglog.cpp
)

target_include_directories(fasteda_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(fasteda_tests GTest::gtest_main)
target_compile_definitions(fasteda_tests PRIVATE FASTEDA_STANDALONE)

include(GoogleTest)
gtest_discover_tests(fasteda_tests)