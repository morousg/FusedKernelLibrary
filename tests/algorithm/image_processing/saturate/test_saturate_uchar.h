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
#include <gtest/gtest.h>

using namespace fk;
 
TEST(SaturateStaticPolymorphism, UCharToUChar) {
  using I = uchar;
  using O = uchar;
  using st = SaturateCastBase<I, O>;
  
  EXPECT_EQ(st::exec(0), static_cast<int>(0));                 // lower bound
  EXPECT_EQ(st::exec(UCHAR_MAX / 2), static_cast<uchar>(UCHAR_MAX / 2)); // in range
  EXPECT_EQ(st::exec(UCHAR_MAX), static_cast<uchar>(UCHAR_MAX));         // Upper bound
}


TEST(SaturateStaticPolymorphism, UCharToFloat) {
  using I = uchar;
  using O = float;
  using st = SaturateCastBase<I, O>;  
  EXPECT_EQ(std::abs(st::exec(0) - (0.0f)) < 1e-1, true);                                // lower bound
  EXPECT_EQ(st::exec(UCHAR_MAX / 2), static_cast<float>(UCHAR_MAX / 2));                 // in range
  EXPECT_EQ(std::abs(st::exec(UCHAR_MAX) - static_cast<float>(UCHAR_MAX)) < 1e-1, true); // upper bound
}

TEST(SaturateStaticPolymorphism, UCharToInt) {
  using I = uchar;
  using O = int;
  using st = SaturateCastBase<I, O>;
  
  EXPECT_EQ(st::exec(0), static_cast<int>(0));                 // lower bound
  EXPECT_EQ(st::exec(UCHAR_MAX / 2), static_cast<int>(UCHAR_MAX / 2)); // in range
  EXPECT_EQ(st::exec(UCHAR_MAX), static_cast<int>(UCHAR_MAX));         // Upper bound
}

TEST(SaturateStaticPolymorphism, UCharToLongLong) {
  using I = uchar;
  using O = longlong;
  using st = SaturateCastBase<I, O>;  
  EXPECT_EQ(st::exec(0), 0);                                        // lower bound
  EXPECT_EQ(st::exec(UCHAR_MAX/2), static_cast<longlong>(UCHAR_MAX/2)); // in range
  EXPECT_EQ(st::exec(UCHAR_MAX), static_cast<longlong>(UCHAR_MAX)); // Upper bound
}

TEST(SaturateStaticPolymorphism, UCharToULongLong) {
  using I = uchar;
  using O = ulonglong;
  using st = SaturateCastBase<I, O>;  
  EXPECT_EQ(st::exec(0), 0);                                         // Lower bound
  EXPECT_EQ(st::exec(UCHAR_MAX/2), static_cast<ulonglong>(UCHAR_MAX/2)); // in range
  EXPECT_EQ(st::exec(UCHAR_MAX), static_cast<ulonglong>(UCHAR_MAX)); // Upper bound
}

TEST(SaturateStaticPolymorphism, UCharToUShort) {
  using I = uchar;
  using O = ushort;
  using st = SaturateCastBase<I, O>;
  
  EXPECT_EQ(st::exec(0), 0);                                      // lower bound
  EXPECT_EQ(st::exec(UCHAR_MAX / 2), static_cast<ushort>(UCHAR_MAX/2)); // in rnage
  EXPECT_EQ(st::exec(UCHAR_MAX), static_cast<ushort>(UCHAR_MAX)); // Upper bound
}

TEST(SaturateStaticPolymorphism, UCharToULong) {
  using I = uchar;
  using O = ulong;
  using st = SaturateCastBase<I, O>;  
  EXPECT_EQ(st::exec(0), 0);                                     // lower bound
  EXPECT_EQ(st::exec(UCHAR_MAX / 2), static_cast<ulong>(UCHAR_MAX / 2)); // in rnage
  EXPECT_EQ(st::exec(UCHAR_MAX), static_cast<ulong>(UCHAR_MAX));          // Upper bound
}
// You can add more tests for other type combinations as needed.
int launch() { return RUN_ALL_TESTS(); };