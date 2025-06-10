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

TEST(SaturateStaticPolymorphism, UIntToUChar) {
  using I = uint;
  using O = uchar;
  using st = SaturateCastBase<I, O>;
  EXPECT_EQ(std::numeric_limits<I>::min(), std::numeric_limits<O>::min());               // Lower bound
  EXPECT_EQ(st::exec(std::numeric_limits<I>::max() / 2), std::numeric_limits<O>::max()); // In range
  EXPECT_EQ(st::exec(std::numeric_limits<I>::max()),std::numeric_limits<O>::max() ); // In range
}

TEST(SaturateStaticPolymorphism, UIntToSChar) {
  using I = uint;
  using O = schar;
  using st = SaturateCastBase<I, O>;
  EXPECT_EQ(std::numeric_limits<I>::min(), std::numeric_limits<O>::min());               // Lower bound
  EXPECT_EQ(st::exec(std::numeric_limits<I>::max() / 2), std::numeric_limits<schar>::max()); // In range
  EXPECT_EQ(st::exec(std::numeric_limits<I>::max()), std::numeric_limits<schar>::max());     // In range
}
 

TEST(SaturateStaticPolymorphism, UIntToUInt) {
  using I = uint;
  using O = uint;
  using st = SaturateCastBase<I, O>;
  EXPECT_EQ(st::exec(0), static_cast<uint>(0));                // Lower bound
  EXPECT_EQ(st::exec(UINT_MAX / 2), static_cast<uint>(UINT_MAX/2)); // In range
  EXPECT_EQ(st::exec(UINT_MAX), static_cast<uint>(UINT_MAX));   // Upper bound
}

TEST(SaturateStaticPolymorphism, UIntToInt) {
  using I = uint;
  using O = uint;
  using st = SaturateCastBase<I, O>;
  EXPECT_EQ(st::exec(0), static_cast<int>(0));                // Lower bound
  EXPECT_EQ(st::exec(UINT_MAX / 2), static_cast<int>(UINT_MAX/2)); // In range
  EXPECT_EQ(st::exec(UINT_MAX), static_cast<int>(UINT_MAX));  // Upper bound
}


TEST(SaturateStaticPolymorphism, UIntToLongLong) {
  using I = uint;
  using O = longlong;
  using st = SaturateCastBase<I, O>;
  EXPECT_EQ(st::exec(0), static_cast<longlong>(0));                // Lower bound
  EXPECT_EQ(st::exec(UINT_MAX / 2), static_cast <longlong>(UINT_MAX/2));     // In range
  EXPECT_EQ(st::exec(UINT_MAX), static_cast<longlong>(UINT_MAX));  // Upper bound
}

TEST(SaturateStaticPolymorphism, UIntToULongLong) {
  using I = uint;
  using O = ulonglong;
  using st = SaturateCastBase<I, O>;
  EXPECT_EQ(st::exec(0), static_cast<ulonglong>(0));                // Lower bound
  EXPECT_EQ(st::exec(UINT_MAX / 2),static_cast <ulonglong>(UINT_MAX/2));                             // In range
  EXPECT_EQ(st::exec(UINT_MAX), static_cast<ulonglong>(UINT_MAX));        // Upper bound
}

TEST(SaturateStaticPolymorphism, UIntToUShort) {
  using I = uint;
  using O = ushort;
  using st = SaturateCastBase<I, O>;
  EXPECT_EQ(st::exec(0), static_cast<ushort>(0));                       // Lower bound
  EXPECT_EQ(st::exec(UINT_MAX / 2), static_cast<ushort>(UINT_MAX / 2)); // In range
  EXPECT_EQ(st::exec(UINT_MAX), static_cast<ushort>(UINT_MAX/2));         // Upper bound
}

TEST(SaturateStaticPolymorphism, UIntToShort) {
  using I = uint;
  using O = short;
  using st = SaturateCastBase<I, O>;
  EXPECT_EQ(st::exec(0), static_cast<short>(0));                       // Lower bound
  EXPECT_EQ(st::exec(UINT_MAX / 2), static_cast<short>(SHRT_MAX)); // In range
  EXPECT_EQ(st::exec(UINT_MAX), static_cast<short>(SHRT_MAX));         // Upper bound
}


TEST(SaturateStaticPolymorphism, UCharToFloat) {
  using I = uchar;
  using O = float;
  using st = SaturateCastBase<I, O>;
  EXPECT_EQ(std::abs(st::exec(0) - (0.0f)) < 1e-1, true);                                // lower bound
  EXPECT_EQ(st::exec(UCHAR_MAX / 2), static_cast<float>(UCHAR_MAX / 2));                 // in range
  EXPECT_EQ(std::abs(st::exec(UCHAR_MAX) - static_cast<float>(UCHAR_MAX)) < 1e-1, true); // upper bound
}

  
// You can add more tests for other type combinations as needed.
int launch() { return RUN_ALL_TESTS(); };