#pragma once
#include "xetla.hpp"

using namespace gpu::xetla;

template <typename dtype_a, typename dtype_b, typename dtype_c,
        typename dtype_scale, typename dtype_zero_pt, typename dtype_table,
        typename dtype_acc, uint32_t wg_m, uint32_t wg_n, uint32_t sg_m,
        uint32_t sg_n, uint32_t sg_k, uint32_t l3_kslicing,
        uint32_t slm_kslicing, uint32_t dequant_s>
struct dequantize_gemm_test_func {
    using tile_shape = group::tile_shape_t<wg_n, wg_m, sg_n, sg_m>;
    static constexpr uint32_t periodic_sync_interval = 1;
    static constexpr uint32_t prefetch_distance = 3;

    using mem_desc_a_t
            = mem_desc_t<dtype_a, mem_layout::row_major, mem_space::global>;
    using mem_desc_b_t
            = mem_desc_t<dtype_b, mem_layout::row_major, mem_space::global>;
    using mem_desc_c_t
            = mem_desc_t<dtype_c, mem_layout::row_major, mem_space::global>;
    using mem_desc_scale_t
            = mem_desc_t<dtype_scale, mem_layout::row_major, mem_space::global>;
    using mem_desc_zero_pt_t = mem_desc_t<dtype_zero_pt, mem_layout::row_major,
            mem_space::global>;
    using mem_desc_table_t
            = mem_desc_t<dtype_table, mem_layout::row_major, mem_space::global>;

    using compute_attr = group::compute_attr_t<fp16, fp16, dtype_acc>;
    using perf_tuning_knob = group::perf_tuning_knob_t<sg_k, prefetch_distance,
            periodic_sync_interval>;

    using compute_policy
            = group::compute_policy_dequantize_matB_xmx<compute_attr,
                    perf_tuning_knob, dtype_scale, dtype_zero_pt, dtype_table,
                    dequant_s, gpu_arch::Xe>;
    using brgemm_t = group::brgemm_t<compute_policy, tile_shape, mem_desc_a_t,
            mem_desc_b_t>;

    using work_group_t = typename brgemm_t::work_group_t;
    static constexpr uint32_t work_group_size = work_group_t::size;
    using brgemm_args_t = typename brgemm_t::arguments_t;
    using matAcc_t = typename brgemm_t::matAcc_t;

    using update_method = typename std::conditional<(l3_kslicing > 1),
            result_reduce_sum, result_overwrite>::type;
    using epilogue_t = group::epilogue_t<
            group::epilogue_policy_tile_op<subgroup::chained_tile_op_t<>,
                    update_method, gpu_arch::Xe>,
            tile_shape, mem_desc_c_t>;

    static_assert(l3_kslicing == 1
                    || std::is_same<remove_const_t<dtype_c>, float>::value
                    || std::is_same<remove_const_t<dtype_c>, int>::value,
            "for l3_kslicing > 1, current we only support float or "
            "int for matC");

    using epilogue_args_t = typename epilogue_t::arguments_t;

    using kslicing_t = group::cooperative_reduce_t<reduce_op::sum, tile_shape,
            matAcc_t, slm_kslicing, gpu_arch::Xe>;
    using mat_slice_t = typename kslicing_t::mat_slice_t;

    static constexpr uint32_t brgemm_nbarr_count = brgemm_t::barrier_count;
    static constexpr uint32_t brgemm_slm_size = brgemm_t::slm_size;

    static constexpr uint32_t epilogue_nbarr_count = epilogue_t::barrier_count;
    static constexpr uint32_t epilogue_slm_size = epilogue_t::slm_size;

    static constexpr uint32_t kslicing_nbarr_count = kslicing_t::barrier_count;
    static constexpr uint32_t kslicing_slm_size = kslicing_t::slm_size;

    static constexpr uint32_t barrier_count = brgemm_nbarr_count * slm_kslicing
            + kslicing_nbarr_count + epilogue_nbarr_count * slm_kslicing;
    static constexpr uint32_t slm_size = brgemm_slm_size * slm_kslicing
            + kslicing_slm_size + epilogue_slm_size * slm_kslicing;
    static_assert(barrier_count <= 32,
            "The named_barrier count should be less than 32!");
    static_assert(slm_size <= (128 * 1024),
            "The local memory size should be less than 128KB!");

    static const char *func_name() { return "dequantize_gemm_test_func"; }

    static inline void run(xetla_exec_item<3> &ei, dtype_a *A, dtype_b *B,
            dtype_c *C, uint32_t mat_m, uint32_t mat_n, uint32_t mat_k,
            uint32_t lda, uint32_t ldb, uint32_t ldc, dtype_scale *scale_ptr,
            uint32_t scale_x, uint32_t scale_y, uint32_t ld_scale,
            dtype_zero_pt *zero_pt_ptr, uint32_t zero_pt_x, uint32_t zero_pt_y,
            uint32_t ld_zero_pt, dtype_table *table_ptr) {

        //        XETLA_PRINT<work_group_size>();
        work_group_t g(ei.get_local_linear_id() % work_group_size);
        uint32_t wg_id = ei.get_local_linear_id() / work_group_size;
        int start_n = ei.get_group(2) * wg_n;
        int start_m = ei.get_group(1) * wg_m;
        int start_k = 0;
        uint32_t wg_tile_k = mat_k;
        uint32_t boundary_k = wg_tile_k;
        if constexpr (l3_kslicing > 1) {
            wg_tile_k = (wg_tile_k + l3_kslicing - 1) / l3_kslicing;
            start_k = start_k + ei.get_group(0) * wg_tile_k;
            boundary_k = (start_k + wg_tile_k) > boundary_k
                    ? boundary_k
                    : (start_k + wg_tile_k);
        }
        if constexpr (slm_kslicing > 1) {
            wg_tile_k = (wg_tile_k + slm_kslicing - 1) / slm_kslicing;
            start_k = start_k + wg_id * wg_tile_k;
            boundary_k = (start_k + wg_tile_k) > boundary_k
                    ? boundary_k
                    : (start_k + wg_tile_k);
        }

        int start_x_scale = start_n;
        int start_y_scale = start_k / brgemm_t::dequant_s;

        int start_x_zero_pt = start_n / 2;
        int start_y_zero_pt = start_k / brgemm_t::dequant_s;

        uint32_t brgemm_slm_base = 0;
        uint32_t brgemm_nbarr_base = 0;
        if constexpr (slm_kslicing > 1) {
            brgemm_slm_base = wg_id * brgemm_slm_size;
            brgemm_nbarr_base = wg_id * brgemm_nbarr_count;
        }
        uint32_t kslicing_slm_base = slm_kslicing * brgemm_slm_size;
        uint32_t kslicing_nbarr_base = slm_kslicing * brgemm_nbarr_count;
        uint32_t epilogue_slm_base = kslicing_slm_base + kslicing_slm_size;
        uint32_t epilogue_nbarr_base
                = kslicing_nbarr_base + kslicing_nbarr_count;

        uint32_t inner_loop_count = (wg_tile_k + sg_k - 1) / sg_k;
        mem_desc_a_t mem_desc_a(
                {A}, {boundary_k, mat_m, lda}, {start_k, start_m});
        mem_desc_b_t mem_desc_b(
                {B}, {mat_n / 2, boundary_k, ldb}, {int(start_n / 2), start_k});

        mem_desc_scale_t mem_desc_scale({scale_ptr},
                {scale_x, scale_y, ld_scale}, {start_x_scale, start_y_scale});
        mem_desc_zero_pt_t mem_desc_zero_pt({zero_pt_ptr},
                {zero_pt_x, zero_pt_y, ld_zero_pt},
                {start_x_zero_pt, start_y_zero_pt});
        mem_desc_table_t mem_desc_table({table_ptr}, {16, 1, 16}, {0, 0});

        matAcc_t matAcc;
        matAcc.init(0);
        brgemm_t brgemm;
        brgemm_args_t brgemm_args(mem_desc_a, mem_desc_b, inner_loop_count,
                mem_desc_scale, mem_desc_zero_pt, mem_desc_table);
        brgemm(g, matAcc, brgemm_args, brgemm_slm_base, brgemm_nbarr_base);

        kslicing_t kslicing(wg_id);
        mat_slice_t mat_slice;
        kslicing(g, mat_slice, matAcc, kslicing_slm_base, kslicing_nbarr_base);

        int32_t coop_offset_n = kslicing.coop_id_x * mat_slice_t::tile_size_x;
        int32_t coop_offset_m = kslicing.coop_id_y * mat_slice_t::tile_size_y;
        mem_desc_c_t mem_desc_c({C}, {mat_n, mat_m, ldc},
                {start_n + coop_offset_n, start_m + coop_offset_m});
        epilogue_t epilogue;
        epilogue_args_t epilogue_args {};
        epilogue(g, mat_slice, mem_desc_c, epilogue_args, epilogue_slm_base,
                epilogue_nbarr_base);
    }
};
