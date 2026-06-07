cmake_minimum_required(VERSION 3.21)
project(fasteda VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Release flags ────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(
    /O2          # Speed optimize
    /W3          # Warning level 3 (not overly pedantic)
    /wd4996      # Silence: 'fopen' unsafe — we handle file safety ourselves
    /wd4834      # Silence: discarding [[nodiscard]] return — intentional
    /D_CRT_SECURE_NO_WARNINGS
  )
else()
  add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter)
endif()

# ── Python ───────────────────────────────────────────────────────
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Threads REQUIRED)

# ── nanobind ─────────────────────────────────────────────────────
# Try system nanobind first (installed via pip in CI)
# Fall back to submodule
find_package(nanobind CONFIG QUIET)

if(NOT nanobind_FOUND)
  message(STATUS "nanobind not found via find_package, using submodule")
  add_subdirectory(extern/nanobind)
endif()

# ── fasteda C++ sources ──────────────────────────────────────────
set(FASTEDA_SOURCES
    src/core/stats_engine.cpp
    src/core/stream_reader.cpp
    src/core/profile_builder.cpp
    src/core/report_serializer.cpp
    src/core/arrow_profiler.cpp
    src/bindings/bindings.cpp
)

# ── nanobind Python extension ────────────────────────────────────
nanobind_add_module(fasteda_core
    STABLE_ABI    # builds one wheel for all Python 3.x
    NB_STATIC     # static link nanobind runtime
    ${FASTEDA_SOURCES}
)

target_include_directories(fasteda_core PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# ── Install ──────────────────────────────────────────────────────
install(TARGETS fasteda_core
    LIBRARY DESTINATION .
)

target_link_libraries(fasteda_core PRIVATE Threads::Threads)

# ── Standalone bench binary (optional, local only) ───────────────
if(NOT SKBUILD)
  add_executable(fasteda_bench
      src/core/stats_engine.cpp
      src/core/stream_reader.cpp
      src/core/profile_builder.cpp
      benchmarks/bench_main.cpp
  )
  target_include_directories(fasteda_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_definitions(fasteda_bench PRIVATE FASTEDA_STANDALONE)
endif()