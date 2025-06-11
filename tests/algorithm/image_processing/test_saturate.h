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

template <typename I, typename O> FK_HOST_DEVICE_CNST bool executeTestSaturateCast(const I& input) {
  static_assert((cn<I>) == (cn<O>), "SaturateCast only accepts I and O types with the same number of channels.");

  using CompType = decltype((make_set<I>(0) == make_set<O>(0)));
  constexpr I inputValue = input;

  constexpr CompType inputGrEqMax = (inputValue >= maxValue<O>);
  // If inputGrEqMax = true, then SaturateCast(inputValue) = O maxValue<O>
  // |-----------0-----I--I--|
  // |-----------0-----O-----|
  // Else, SaturateCast(inputValue) = O inputValue
  // |-----------0---I-------|
  // |-----------0-----O-----|
  constexpr CompType inputLsEqMin = (inputValue <= minValue<O>);
  // If inputLsEqMin = true, then SaturateCast(inputValue) = O minValue<O>
  // |-I--I------0-----------|
  // |----O------0-----------|
  // Else, SaturateCast(inputValue) = O minValue<I>
  // |-------I---0-----------|
  // |----O------0-----------|

  using SaturareCast = SaturateCast<I, O>;
  constexpr I input_min_range = minValue<I>;
  constexpr I input_max_range = maxValue<I>;

  constexpr O output_min_range = minValue<O>;
  constexpr O output_max_range = maxValue<O>;

  if constexpr (inputGrEqMax) {
    //Expected result maxValue<O>
    return VectorAnd<CompType>::exec(SaturareCast::exec(input) == maxValue<O>);
  } else if constexpr (inputLsEqMin) {
    //Expected result minValue<O>
    return VectorAnd<CompType>::exec(SaturareCast::exec(input) == minValue<O>);
  } else {
    //Expected result inputValue
    return VectorAnd<CompType>::exec(SaturareCast::exec(input) == Cast<I,O>::exec(inputValue));
  }
}

template <typename I, typename O> FK_HOST_DEVICE_CNST bool executeFullTestSaturateCast() {
  static_assert((cn<I>) == (cn<O>), "SaturateCast only accepts I and O types with the same number of channels.");
  using CompType = decltype(std::declval<I>() == std::declval<O>());

  constexpr CompType inputTypeGrMax = (maxValue<I> > maxValue<O>);
  constexpr CompType inputTypeLsMin = (minValue<I> < minValue<O>);

  // TODO: generate test values according to I and O types
  // If maxValue<I> > maxValue<O> && minValue<I> < minValue<O>
  // Then we need:
  // 1. A value of type I greater than maxValue<O> to test the saturation to maxValue<O>

  // 2. A value of type I less than minValue<O> to test the saturation to minValue<O>
  // 3. A value of type I between minValue<O> and maxValue<O> to test the normal cast (we keep the same value)
  // 4. A value of type I equal to minValue<O> to test the saturation to minValue<O>
  // 5. A value of type I equal to maxValue<O> to test the saturation to maxValue<O>
  if constexpr (inputTypeGrMax && inputTypeLsMin) {
    //

  } else if constexpr (!inputTypeGrMax && inputTypeLsMin) {

  } else if constexpr (inputTypeGrMax && !inputTypeLsMin) {

  } else if constexpr (!inputTypeGrMax && !inputTypeLsMin) {

  } else {

    static_assert(false, "Invalid combination of input and output types for SaturateCast.");
  }
}

TEST(TestSaturateCast, UInt1ToUInt1) {
  const bool result = executeFullTestSaturateCast<uint1, uint1>();
  EXPECT_EQ(result, true);
}

// You can add more tests for other type combinations as needed.
int launch() { return RUN_ALL_TESTS(); };