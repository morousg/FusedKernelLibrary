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

TEST(SaturateStaticPolymorphism, SCharToSChar) {
  using I = schar;
  using O = schar;
  using st = SaturateCastBase<I, O>;  
  EXPECT_EQ(st::exec(SCHAR_MIN) , static_cast <schar>(SCHAR_MIN)); // Lower bound
  EXPECT_EQ(st::exec(0),0);  // In range
  EXPECT_EQ(st::exec(SCHAR_MAX) , static_cast<schar>(SCHAR_MAX));       // In range
}

TEST(SaturateStaticPolymorphism, SCharToFloat) {
  using I = schar;
  using O = float;
  using st = SaturateCastBase<I, O>;  
  EXPECT_EQ(std::abs(st::exec(SCHAR_MIN) - static_cast <float>(SCHAR_MIN)) < 1e-1, true); // Lower bound
  EXPECT_EQ(std::abs(st::exec(0) - (0.0f)) < 1e-1, true);  // In range
  EXPECT_EQ(std::abs(st::exec(SCHAR_MAX) - static_cast<float>(SCHAR_MAX)) < 1e-1, true);       // In range
}

 
TEST(SaturateStaticPolymorphism, SCharToInt) {
  using I = schar;
  using O = int;
  using st = SaturateCastBase<I, O>;  
  EXPECT_EQ(st::exec(SCHAR_MIN), static_cast<int>(SCHAR_MIN)); // Lower bound
  EXPECT_EQ(st::exec(0),  static_cast<int>(0)); // In range
  EXPECT_EQ(st::exec(SCHAR_MAX),  static_cast<int>(SCHAR_MAX)); // Upper bound  
  
}

TEST(SaturateStaticPolymorphism, SCharToLongLong) {
  using I = schar;
  using O = longlong;
  using st = SaturateCastBase<I, O>;
  EXPECT_EQ(st::exec(SCHAR_MIN), static_cast<longlong>(SCHAR_MIN)); // Lower bound
  EXPECT_EQ(st::exec(0), 0);       // In range
  EXPECT_EQ(st::exec(SCHAR_MAX), static_cast<longlong>(SCHAR_MAX)); // Upper bound
}

TEST(SaturateStaticPolymorphism, SCharToULongLong) {
  using I = schar;
  using O = ulonglong;
  using st = SaturateCastBase<I, O>;
  EXPECT_EQ(st::exec(SCHAR_MIN), static_cast<ulonglong>(SCHAR_MIN)); // Lower bound
  EXPECT_EQ(st::exec(0), 0);                                        // In range
  EXPECT_EQ(st::exec(SCHAR_MAX), static_cast<ulonglong>(SCHAR_MAX)); // Upper bound
}

TEST(SaturateStaticPolymorphism, SCharToUShort) {
  using I = schar;
  using O = ushort;
  using st = SaturateCastBase<I, O>;
  EXPECT_EQ(st::exec(SCHAR_MIN), 0); // Lower bound
  EXPECT_EQ(st::exec(0), 0);                                         // In range
  EXPECT_EQ(st::exec(SCHAR_MAX), static_cast<ushort>(SCHAR_MAX)); // Upper bound
}

TEST(SaturateStaticPolymorphism, SCharToULong) {
  using I = schar;
  using O = ulong;
  using st = SaturateCastBase<I, O>;
  EXPECT_EQ(st::exec(SCHAR_MIN), static_cast<ulong>(SCHAR_MIN));  // Lower bound
  EXPECT_EQ(st::exec(0), 0);                                      // In range
  EXPECT_EQ(st::exec(SCHAR_MAX), static_cast<ulong>(SCHAR_MAX));  // Upper bound
}
 
// You can add more tests for other type combinations as needed.
int launch() { return RUN_ALL_TESTS(); };