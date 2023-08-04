#pragma once
#include "xetla.hpp"

using namespace gpu::xetla;

template <typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_acc, uint32_t wg_m, uint32_t wg_n, uint32_t sg_m,
        uint32_t sg_n, uint32_t sg_k, mem_layout layout_a, mem_layout layout_b,
        uint32_t l3_kslicing, uint32_t slm_kslicing>
struct fpu_gemm_test_func {
    using tile_shape = group::tile_shape_t<wg_n, wg_m, sg_n, sg_m>;
    static constexpr uint32_t periodic_sync_interval = 0;
    static constexpr uint32_t prefetch_distance = 1;

    using mem_desc_input_a = mem_desc_t<dtype_a, layout_a, mem_space::global>;
    using mem_desc_input_b = mem_desc_t<dtype_b, layout_b, mem_space::global>;
    using mem_desc_output_c
            = mem_desc_t<dtype_c, mem_layout::row_major, mem_space::global>;

    using compute_attr = group::compute_attr_t<dtype_acc, dtype_acc, dtype_acc>;
    using perf_tuning_knob = group::perf_tuning_knob_t<sg_k, prefetch_distance,
            periodic_sync_interval>;
    using compute_policy = group::compute_policy_default_fpu<compute_attr,
            perf_tuning_knob, gpu_arch::Xe>;

    using brgemm_t = group::brgemm_t<compute_policy, tile_shape,
            mem_desc_input_a, mem_desc_input_b>;

    using update_method = typename std::conditional<(l3_kslicing > 1),
            result_reduce_sum, result_overwrite>::type;
    using epilogue_t = group::epilogue_t<
            group::epilogue_policy_tile_op<subgroup::chained_tile_op_t<>,
                    update_method, gpu_arch::Xe>,
            tile_shape, mem_desc_output_c>;
    using gemm_op_t
            = kernel::gemm_t<kernel::dispatch_policy_kslicing<l3_kslicing,
                                     slm_kslicing, gpu_arch::Xe>,
                    brgemm_t, epilogue_t>;
    static constexpr uint32_t barrier_count = gemm_op_t::get_barrier_count();
    static constexpr uint32_t slm_size = gemm_op_t::get_slm_size();

    static const char *func_name() { return "fpu_gemm_test_func"; }

    static inline void run(xetla_exec_item<3> &ei, dtype_a *A, dtype_b *B,
            dtype_c *C, uint32_t mat_m, uint32_t mat_n, uint32_t mat_k) {
        typename gemm_op_t::arguments_t arg(mat_m, mat_k, mat_n, A,
                layout_a == mem_layout::col_major ? mat_m : mat_k, B,
                layout_b == mem_layout::col_major ? mat_k : mat_n, C, mat_n);
        gemm_op_t gemm_op;
        gemm_op(ei, arg);
    }
};
