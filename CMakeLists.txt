cmake_minimum_required(VERSION 3.16)
project(OlapEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Checking for compiler")
if(MSVC)
    # Enable AVX2
    add_compile_options(/arch:AVX2)
    
    add_compile_options(/MP) 
    
    message(STATUS "MSVC compiler detected. Enabling /arch:AVX2 and /MP (multi-core build)")
else()
    # Enable AVX2
    add_compile_options(-mavx2)
    
    message(STATUS "GCC/Clang compiler detected. Enabling -mavx2")
    message(STATUS "For parallel builds on Linux, run 'make -j8' or 'ninja'")
endif()


find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

message(STATUS "Found Apache Arrow version: ${Arrow_VERSION}")

# Collect all source files except main files (to avoid multiple main() definitions)
file(GLOB_RECURSE ALL_SOURCES "src/*.cpp")

# Exclude main files from the common sources
set(COMMON_SOURCES ${ALL_SOURCES})
list(FILTER COMMON_SOURCES EXCLUDE REGEX "src/main\\.cpp$")
list(FILTER COMMON_SOURCES EXCLUDE REGEX "src/benchmark_uniform\\.cpp$")
list(FILTER COMMON_SOURCES EXCLUDE REGEX "src/benchmark_skewed\\.cpp$")
list(FILTER COMMON_SOURCES EXCLUDE REGEX "src/benchmark_normal\\.cpp$")

# Main engine executable (original benchmark)
add_executable(engine ${COMMON_SOURCES} "src/main.cpp")

# Uniform data benchmark executable
add_executable(benchmark_uniform ${COMMON_SOURCES} "src/benchmark_uniform.cpp")

# Skewed data benchmark executable
add_executable(benchmark_skewed ${COMMON_SOURCES} "src/benchmark_skewed.cpp")

# Normal distribution data benchmark executable
add_executable(benchmark_normal ${COMMON_SOURCES} "src/benchmark_normal.cpp")

# Configure all executables
foreach(TARGET_NAME engine benchmark_uniform benchmark_skewed benchmark_normal)
    # pch
    target_precompile_headers(${TARGET_NAME}
        PRIVATE
            "src/pch.h"
    )

    target_link_libraries(${TARGET_NAME}
        PUBLIC
            "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Arrow::arrow_static,Arrow::arrow_shared>"
            "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Parquet::parquet_static,Parquet::parquet_shared>"
    )

    target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
endforeach()
