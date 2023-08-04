/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "common.hpp"
#include "kernel_func.hpp"
#include "utils/utils.hpp"
#include <gtest/gtest.h>

using namespace cl::sycl;

std::string esimd_compile_string
        = " -vc-codegen -doubleGRF  -Xfinalizer ' -printregusage -noLocalSplit "
          "-enableBCR -nolocalra  ' ";

template <typename T>
class sgemm_test : public ::testing::Test {};
TYPED_TEST_SUITE_P(sgemm_test);
TYPED_TEST_P(sgemm_test, esimd) {
    gemm_exec<TypeParam, typename TypeParam::data_type_a,
            typename TypeParam::data_type_b, typename TypeParam::data_type_c,
            typename TypeParam::data_type_acc, input_buffer_init,
            result_validate, sgemm_func>(TypeParam::mat_m, TypeParam::mat_n,
            TypeParam::mat_k, esimd_compile_string);
}
REGISTER_TYPED_TEST_SUITE_P(sgemm_test, esimd);
using tests = ::testing::Types<Test1, Test2, Test3, Test4, Test5, Test6, Test7,
        Test8, Test1_fp16, Test2_fp16, Test3_fp16, Test4_fp16, Test5_fp16,
        Test6_fp16, Test7_fp16, Test8_fp16>;
INSTANTIATE_TYPED_TEST_SUITE_P(sgemm_test_suite, sgemm_test, tests);
