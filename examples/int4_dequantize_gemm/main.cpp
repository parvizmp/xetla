#include "kernel_func.hpp"
#include "test.hpp"
#include "tests/utils/profiling.hpp"
#include "tests/utils/utils.hpp"

template <typename data_type_a, typename data_type_b, typename data_type_c,
        typename data_type_acc = float>
int gemm_result_validate(data_type_a *A, data_type_b *B, data_type_c *C,
        uint32_t m, uint32_t k, uint32_t n,
        mem_layout mem_layout_a_ = mem_layout::row_major,
        mem_layout mem_layout_b_ = mem_layout::row_major) {
    buff_cmp::buff_vals<data_type_c> data(C, m, n, n);
    std::vector<data_type_acc> gold_C(m * n, 0);
    get_gemm_gold<data_type_a, data_type_b, data_type_acc>(
            m, n, k, mem_layout_a_, mem_layout_b_, A, B, gold_C.data());
    buff_cmp::buff_vals<data_type_c, data_type_acc> other(
            gold_C.data(), m, n, n);

    bool result
            = buff_cmp::xetla_buff_cmp(data, other, "basic_gemm validation");

    std::cout << (!result ? "FAILED\n" : "PASSED\n");
    return result ? 0 : 1;
}

template <class Test>
void dequantize_gemm_run(int iter) {
    //Accept incoming parameters
    constexpr size_t matrix_m = Test::mat_m;
    constexpr size_t matrix_n = Test::mat_n;
    constexpr size_t matrix_k = Test::mat_k;
    constexpr uint32_t l3_kslicing = Test::l3_kslicing;
    constexpr uint32_t slm_kslicing = Test::slm_kslicing;

    constexpr size_t wg_tile_m = Test::wg_m;
    constexpr size_t wg_tile_n = Test::wg_n;
    constexpr size_t sg_tile_m = Test::sg_m;
    constexpr size_t sg_tile_n = Test::sg_n;
    constexpr size_t sg_tile_k = Test::sg_k;
    constexpr size_t dequant_s = Test::dequant_s;
    using data_type_a = typename Test::data_type_a;
    using data_type_b = typename Test::data_type_b;
    using data_type_c = typename Test::data_type_c;
    using data_type_zero_pt = bit4x2;
    using data_type_scale = fp16;
    using data_type_table = fp16;
    using data_type_acc = float;
    constexpr uint32_t num_buffer = Test::num_buffer;

    uint16_t lookup_table[16] = {0x0000, 0x3c00, 0x4000, 0x4400, 0x4800, 0x4c00,
            0x5000, 0x5400, 0x0000, 0xbc00, 0xc000, 0xc400, 0xc800, 0xcc00,
            0xd000, 0xd400};
    constexpr size_t size_a = matrix_m * matrix_k;
    constexpr size_t size_b = matrix_k * matrix_n / 2;

    constexpr size_t size_scale_m = matrix_k / dequant_s;
    constexpr size_t size_scale_n = matrix_n;
    constexpr size_t size_scale = size_scale_m * size_scale_n;

    constexpr size_t size_zero_pt_m = matrix_k / dequant_s;
    constexpr size_t size_zero_pt_n = matrix_n / 2;
    constexpr size_t size_zero_pt = size_zero_pt_m * size_zero_pt_n;

    constexpr size_t size_c = matrix_m * matrix_n;
    uint32_t lda = matrix_k;
    //bit4x2
    uint32_t ldb = matrix_n / 2;
    uint32_t ldc = matrix_n;
    uint32_t ld_scale = size_scale_n;
    uint32_t ld_zero_pt = size_zero_pt_n;

    //Turn on the enable_profiling property to facilitate subsequent profiling
    sycl::property_list properties {sycl::property::queue::enable_profiling()};
    auto Queue = queue(properties);
    auto Context = Queue.get_info<info::queue::context>();
    auto Device = Queue.get_info<info::queue::device>();

    std::cout << "Running on " << Device.get_info<info::device::name>() << "\n";

    //Define and initialize the data required for the calculation
    data_type_a *A = static_cast<data_type_a *>(malloc_shared(
            size_a * sizeof(data_type_a) * num_buffer, Device, Context));
    data_type_b *B = static_cast<data_type_b *>(malloc_shared(
            size_b * sizeof(data_type_b) * num_buffer, Device, Context));
    data_type_c *C = static_cast<data_type_c *>(malloc_shared(
            size_c * sizeof(data_type_c) * num_buffer, Device, Context));
    data_type_scale *scale = static_cast<data_type_scale *>(
            malloc_shared(size_scale * sizeof(data_type_scale) * num_buffer,
                    Device, Context));
    data_type_zero_pt *zero_pt = static_cast<data_type_zero_pt *>(
            malloc_shared(size_zero_pt * sizeof(data_type_zero_pt) * num_buffer,
                    Device, Context));

    data_type_table *table = static_cast<data_type_table *>(
            malloc_shared(16 * sizeof(data_type_table), Device, Context));

    for (unsigned i = 0; i < size_a * num_buffer; ++i) {
        A[i] = random_float();
    }
    for (unsigned i = 0; i < size_b * num_buffer; ++i) {
        B[i] = uint8_t(rand());
    }
    for (unsigned i = 0; i < size_scale * num_buffer; ++i) {
        scale[i] = random_float();
    }
    for (unsigned i = 0; i < size_zero_pt * num_buffer; ++i) {
        zero_pt[i] = uint8_t(rand());
    }
    for (unsigned i = 0; i < 16; ++i) {
        table[i] = (*reinterpret_cast<data_type_table *>(&lookup_table[i]));
    }
    for (unsigned i = 0; i < size_c * num_buffer; ++i) {
        C[i] = 0;
    }
    // here keep the same dim in CM and esimd, diff the index in kernel code
    size_t group_range_m = (matrix_m % wg_tile_m == 0)
            ? matrix_m / wg_tile_m
            : (matrix_m / wg_tile_m) + 1;
    size_t group_range_n = (matrix_n % wg_tile_n == 0)
            ? matrix_n / wg_tile_n
            : (matrix_n / wg_tile_n) + 1;
    size_t subgroup_range_m = (wg_tile_m % sg_tile_m == 0)
            ? wg_tile_m / sg_tile_m
            : (wg_tile_m / sg_tile_m) + 1;
    size_t subgroup_range_n = (wg_tile_n % sg_tile_n == 0)
            ? wg_tile_n / sg_tile_n
            : (wg_tile_n / sg_tile_n) + 1;
    std::cout << "group_num_x: " << group_range_n
              << ", group_num_y: " << group_range_m
              << ", group_num_z: " << l3_kslicing << "\n";
    std::cout << "group_size_x: " << subgroup_range_n
              << ", group_size_y: " << subgroup_range_m
              << ", group_size_z: " << slm_kslicing << std::endl;
    cl::sycl::range<3> GroupRange {l3_kslicing, group_range_m, group_range_n};
    cl::sycl::range<3> LocalRange {
            slm_kslicing, subgroup_range_m, subgroup_range_n};
    cl::sycl::nd_range<3> Range(GroupRange * LocalRange, LocalRange);

    size_t ops = 2 * matrix_m * matrix_n * matrix_k;
    profiling_helper prof("dequantize_gemm", ops, "gflops");

    try {
        for (int i = 0; i < iter; i++) {
            prof.cpu_start();
            auto e_esimd = Queue.submit([&](handler &cgh) {
                //                cgh.use_kernel_bundle(exeBundle);
                cgh.parallel_for<Test>(
                        Range, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                            xetla_exec_item<3> ei(item);
                            using dequantize_gemm_functor
                                    = dequantize_gemm_test_func<data_type_a,
                                            data_type_b, data_type_c,
                                            data_type_scale, data_type_zero_pt,
                                            data_type_table, data_type_acc,
                                            wg_tile_m, wg_tile_n, sg_tile_m,
                                            sg_tile_n, sg_tile_k, l3_kslicing,
                                            slm_kslicing, dequant_s>;
                            constexpr uint32_t barrier_count
                                    = dequantize_gemm_functor::barrier_count;
                            constexpr uint32_t slm_size
                                    = dequantize_gemm_functor::slm_size;

                            xetla_nbarrier_init<barrier_count>();
                            xetla_local_init<slm_size>();

                            dequantize_gemm_functor::run(ei,
                                    A + (i % num_buffer) * size_a,
                                    B + (i % num_buffer) * size_b,
                                    C + (i % num_buffer) * size_c, matrix_m,
                                    matrix_n, matrix_k, lda, ldb, ldc,
                                    scale + (i % num_buffer) * size_scale,
                                    size_scale_n, size_scale_m, ld_scale,
                                    zero_pt + (i % num_buffer) * size_zero_pt,
                                    size_zero_pt_n, size_zero_pt_m, ld_zero_pt,
                                    table);
                        });
            });
            e_esimd.wait();
            prof.cpu_end();
            prof.add_gpu_event(e_esimd);
        }
    } catch (cl::sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        FAIL();
    }

    //performance
    prof.print_profiling_result(profiling_selector::GPU);

    // validation
    int last_itr = (iter - 1) % num_buffer;
    auto last_scale = scale + last_itr * size_scale;
    auto last_zero_pt = zero_pt + last_itr * size_zero_pt;
    auto last_B = B + last_itr * size_b;
    std::vector<fp16> dequantize_b(matrix_k * matrix_n, 0);
    for (int i = 0; i < matrix_k / dequant_s; i++) {
        for (int j = 0; j < matrix_n / 16; j++) {
            int start_in = i * dequant_s * matrix_n / 2 + j * 8;
            int start_out = i * dequant_s * matrix_n + j * 16;
            int start_scale = i * size_scale_n + j * 16;
            int start_zero_pt = i * size_zero_pt_n + j * 8;
            for (int ii = 0; ii < dequant_s; ii++) {
                for (int jj = 0; jj < 8; jj++) {
                    uint8_t data_in = last_B[start_in + ii * matrix_n / 2 + jj];
                    uint8_t data_zero_pt = last_zero_pt[start_zero_pt + jj];

                    int8_t data_0 = int8_t(data_in & 0x0f);
                    int8_t data_1 = int8_t(data_in >> 4);

                    int8_t zero_pt_0 = int8_t((data_zero_pt & 0x0f) + 1);
                    int8_t zero_pt_1 = int8_t((data_zero_pt >> 4) + 1);

                    dequantize_b[start_out + ii * matrix_n + jj * 2]
                            = fp16(data_0 - zero_pt_0)
                            * last_scale[start_scale + jj * 2];
                    dequantize_b[start_out + ii * matrix_n + jj * 2 + 1]
                            = fp16(data_1 - zero_pt_1)
                            * last_scale[start_scale + jj * 2 + 1];
                }
            }
        }
    }

    ASSERT_EQ(0,
            gemm_result_validate(A + last_itr * size_a, dequantize_b.data(),
                    C + last_itr * size_c, matrix_m, matrix_k, matrix_n,
                    Test::layout_a, Test::layout_b));

    free(A, Context);
    free(B, Context);
    free(C, Context);
    free(scale, Context);
    free(table, Context);
    free(zero_pt, Context);
}

template <typename T>
class dequantize_gemm_test : public ::testing::Test {};
TYPED_TEST_SUITE_P(dequantize_gemm_test);

TYPED_TEST_P(dequantize_gemm_test, esimd) {
    dequantize_gemm_run<TypeParam>(ITER);
}

REGISTER_TYPED_TEST_SUITE_P(dequantize_gemm_test, esimd);
using tests = ::testing::Types<qkv, output_proj, ffn1, ffn2, last>;

INSTANTIATE_TYPED_TEST_SUITE_P(
        dequantize_gemm_test_suite, dequantize_gemm_test, tests);
