#include "common/common.hpp"
#include "tests/utils/common.hpp"

using namespace gpu::xetla;
//The number of times the kernel is executed
constexpr int ITER = 100;

class qkv {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 4096 * 3;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 1;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 1;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 32;
    static constexpr size_t num_buffer = 64;
    static constexpr size_t slm_kslicing = 1;
    static constexpr size_t l3_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = fp16;
};

class output_proj {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 64;
    static constexpr size_t num_buffer = 128;
    static constexpr size_t slm_kslicing = 8;
    static constexpr size_t l3_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = fp16;
};

class ffn1 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 16384;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 256;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 32;
    static constexpr size_t num_buffer = 32;
    static constexpr size_t slm_kslicing = 2;
    static constexpr size_t l3_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = fp16;
};

class ffn2 {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_k = 16384;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 64;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 64;
    static constexpr size_t num_buffer = 32;
    static constexpr size_t slm_kslicing = 8;
    static constexpr size_t l3_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = fp16;
};

class last {
public:
    //Extract the parameters required by different test cases
    static constexpr size_t mat_m = 1;
    static constexpr size_t mat_n = 50400;
    static constexpr size_t mat_k = 4096;
    static constexpr size_t wg_m = 8;
    static constexpr size_t wg_n = 512;
    static constexpr size_t sg_m = 8;
    static constexpr size_t sg_n = 16;
    static constexpr size_t sg_k = 32;
    static constexpr size_t num_buffer = 16;
    static constexpr size_t slm_kslicing = 1;
    static constexpr size_t l3_kslicing = 1;
    static constexpr mem_layout layout_a = mem_layout::row_major;
    static constexpr mem_layout layout_b = mem_layout::row_major;
    using data_type_a = fp16;
    using data_type_b = fp16;
    using data_type_c = fp16;
};
