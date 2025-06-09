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



#include <gtest/gtest.h>
#include "fused_kernel/algorithms/image_processing/saturate.h"

using namespace fk;

// Helper function template to test static polymorphism
template <typename I, typename O> void TestSaturateCastBase(I input, O expected) {
  SaturateCastBase<I, O> saturate;
  EXPECT_EQ(saturate.exec(input), expected);
}

TEST(SaturateStaticPolymorphism, IntToUchar) {
  TestSaturateCastBase<int, uchar>(-10, 0);
  TestSaturateCastBase<int, uchar>(0, 0);
  TestSaturateCastBase<int, uchar>(128, 128);
  TestSaturateCastBase<int, uchar>(255, 255);
  TestSaturateCastBase<int, uchar>(300, 255);
}

TEST(SaturateStaticPolymorphism, FloatToUchar) {
  TestSaturateCastBase<float, uchar>(-1.5f, 0);
  TestSaturateCastBase<float, uchar>(0.0f, 0);
  TestSaturateCastBase<float, uchar>(127.9f, 128);
  TestSaturateCastBase<float, uchar>(255.0f, 255);
  TestSaturateCastBase<float, uchar>(300.0f, 255);
}

TEST(SaturateStaticPolymorphism, UcharToSchar) {
  TestSaturateCastBase<uchar, schar>(0, 0);
  TestSaturateCastBase<uchar, schar>(127, 127);
  TestSaturateCastBase<uchar, schar>(128, 127);
  TestSaturateCastBase<uchar, schar>(255, 127);
}

// You can add more tests for other type combinations as needed.