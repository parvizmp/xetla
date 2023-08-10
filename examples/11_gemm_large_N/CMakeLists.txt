set(TARGET gemm_large_N)

#build test
add_executable(${TARGET} gemm_large_N.cpp)
#target_compile_options(${TARGET} PUBLIC -fsycl -gline-tables-only -fdebug-info-for-profiling)