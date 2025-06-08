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

TEST(SaturateCastBaseTest, IntToUchar) {
    SaturateCastBase<int, uchar> saturate;
    EXPECT_EQ(saturate.exec(-10), 0);      // Below range
    EXPECT_EQ(saturate.exec(0), 0);        // Lower bound
    EXPECT_EQ(saturate.exec(128), 128);    // In range
    EXPECT_EQ(saturate.exec(255), 255);    // Upper bound
    EXPECT_EQ(saturate.exec(300), 255);    // Above range
}

TEST(SaturateCastBaseTest, FloatToUchar) {
    SaturateCastBase<float, uchar> saturate;
    EXPECT_EQ(saturate.exec(-1.5f), 0);
    EXPECT_EQ(saturate.exec(0.0f), 0);
    EXPECT_EQ(saturate.exec(127.9f), 128);
    EXPECT_EQ(saturate.exec(255.0f), 255);
    EXPECT_EQ(saturate.exec(300.0f), 255);
}

TEST(SaturateCastBaseTest, UcharToSchar) {
    SaturateCastBase<uchar, schar> saturate;
    EXPECT_EQ(saturate.exec(0), 0);
    EXPECT_EQ(saturate.exec(127), 127);
    EXPECT_EQ(saturate.exec(128), 127);
    EXPECT_EQ(saturate.exec(255), 127);
}
int launch() {

  return RUN_ALL_TESTS();
};

    // Add more tests for other type combinations as needed