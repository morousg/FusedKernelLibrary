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
 
/*
using uchar = unsigned char;
using schar = signed char;
using uint = unsigned int;
using longlong = long long;
using ulonglong = unsigned long long;
using ushort = unsigned short;
using ulong = unsigned long;
*/



/*UCHAR**/


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
/*SCHAR**/

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
 

TEST(SaturateStaticPolymorphism, IntToUchar) {
  using I = int;
  using O = uchar;
  using st = SaturateCastBase<I, O>;
  EXPECT_EQ(st::exec(SCHAR_MIN), static_cast<ulong>(SCHAR_MIN)); // Lower bound

  EXPECT_EQ(st::exec(-10), 0);   // Below range
  EXPECT_EQ(st::exec(0), 0);     // Lower bound
  EXPECT_EQ(st::exec(128), 128); // In range
  EXPECT_EQ(st::exec(255), 255); // Upper bound
  EXPECT_EQ(st::exec(300), 255); // Above range
}
TEST(SaturateStaticPolymorphism, FloatToUchar) {
  using I = float;
  using O = uchar;
  using st = SaturateCastBase<I, O>;
  EXPECT_EQ(st::exec(-1.5f), 0);
  EXPECT_EQ(st::exec(0.0f), 0);
  EXPECT_EQ(st::exec(127.9f), 128);
  EXPECT_EQ(st::exec(255.0f), 255);
  EXPECT_EQ(st::exec(300.0f), 255);
}

// You can add more tests for other type combinations as needed.
int launch() { return RUN_ALL_TESTS(); };