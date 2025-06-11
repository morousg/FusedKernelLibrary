/* Copyright 2025 the Fused Kernel Project Developers

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "fused_kernel/algorithms/image_processing/saturate.h"
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/cuda_vector.h>
#include <fused_kernel/core/utils/type_to_string.h>
#include <fused_kernel/core/utils/vlimits.h>

#include <gtest/gtest.h>
#define ONLY_CPU // Disable CUDA for this test, as it is not needed and can complicate the test environment.
using namespace fk;

template <typename I, typename O>
struct TestData;

#define SATURATE_CAST_TEST_DATA(IN, OU)                                                                                \
  template <> struct TestData<IN, OU> {                                                                                \
    static_assert(cn<IN> == cn<OU>, "Input and output types must have the same number of channels");                   \
    using I = IN;                                                                                                      \
    using O = OU;                                                                                                      \
    static constexpr size_t CH = cn<I>; \
static constexpr size_t NUM_VALS{ NUM_VALUES_##IN##_##OU };                                                                   \
    static constexpr std::array<I, NUM_VALS> inputValues INPUT_VALUES_##IN##_##OU;                                               \
    static constexpr std::array<O, NUM_VALS> output EXPECTED_VALUES_##IN##_##OU;                                       \
  };

#define NUM_VALUES_uint1_uint1 3
#define INPUT_VALUES_uint1_uint1    {minValue<uint1>, maxValue<uint1> / static_cast<uint>(2), maxValue<uint1>}
#define EXPECTED_VALUES_uint1_uint1 {minValue<uint1>, maxValue<uint1> / static_cast<uint>(2), maxValue<uint1>}
SATURATE_CAST_TEST_DATA(uint1, uint1)
    


template <typename TestDataCase, size_t... Idx>
constexpr auto testDataElems(const std::index_sequence<Idx...> &) {
    using I = typename TestDataCase::I;
    using O = typename TestDataCase::O;
    return std::array<O, sizeof...(Idx)>{SaturateCast<I, O>::exec(TestData<I, O>::inputValues[Idx])...};
}

TEST(TestSaturateCast, UInt1ToUInt1) {
  using TestDataCase = TestData<uint1, uint1>;
  constexpr auto generatedValues = testDataElems<TestDataCase>(std::make_index_sequence<TestDataCase::NUM_VALS>{});

  for (int numTest = 0; numTest < TestDataCase::NUM_VALS; ++numTest) {
    const auto generated = toArray(generatedValues[numTest]);
    const auto expected = toArray(TestDataCase::output[numTest]);
    for (int vecPos = 0; vecPos < TestDataCase::CH; ++vecPos) {
      EXPECT_EQ(generated[vecPos], expected[vecPos])
          << "Mismatch at position " << vecPos << " for test case " << numTest << " with I=uint1 and O=uint1";
    }
  }
}

// You can add more tests for other type combinations as needed.
int launch() { return RUN_ALL_TESTS(); };