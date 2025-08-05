/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Grup Mediapro S.L.U

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/core/utils/type_to_string.h>
#include <tests/operation_test_utils.h>

// Test PixelFormatTraits for UYVY
void testUYVYPixelFormatTraits() {
    constexpr int expectedSpace = static_cast<int>(fk::ColorSpace::YUV422);
    constexpr int expectedDepth = static_cast<int>(fk::ColorDepth::p8bit);
    constexpr int expectedCn = 3;
    
    static_assert(fk::PixelFormatTraits<fk::PixelFormat::UYVY>::space == expectedSpace, "UYVY space should be YUV422");
    static_assert(fk::PixelFormatTraits<fk::PixelFormat::UYVY>::depth == expectedDepth, "UYVY depth should be p8bit");
    static_assert(fk::PixelFormatTraits<fk::PixelFormat::UYVY>::cn == expectedCn, "UYVY cn should be 3");
}

// Test IsEven function used in UYVY processing
void testIsEvenFunction() {
    const std::string testName = "Test_IsEven_uint";
    
    constexpr std::array<uint, 4> inputVals{ 0, 1, 2, 3 };
    constexpr std::array<bool, 4> expectedVals{ true, false, true, false };
    
    testCases[testName] = 
        TestCaseBuilder<fk::IsEven<uint>>::build(testName, inputVals, expectedVals);
}

// Test RGB2Gray conversion functionality
void testRGB2GrayConversion() {
    const std::string testName = "Test_RGB2Gray_uchar3_uchar";
    
    // Test with known RGB values
    constexpr std::array<uchar3, 3> inputVals{
        uchar3{255, 0, 0},    // Pure red -> expected ~77 (0.299*255)
        uchar3{0, 255, 0},    // Pure green -> expected ~150 (0.587*255)  
        uchar3{0, 0, 255}     // Pure blue -> expected ~29 (0.114*255)
    };
    
    constexpr std::array<uchar, 3> expectedVals{
        76,    // 0.299*255 ≈ 76.245 -> 76
        150,   // 0.587*255 ≈ 149.685 -> 150 (rounded)
        29     // 0.114*255 ≈ 29.07 -> 29
    };
    
    testCases[testName] = 
        TestCaseBuilder<fk::RGB2Gray<uchar3, uchar>>::build(testName, inputVals, expectedVals);
}

// Test AddOpaqueAlpha functionality
void testAddOpaqueAlpha() {
    const std::string testName = "Test_AddOpaqueAlpha_uchar3_p8bit";
    
    constexpr std::array<uchar3, 2> inputVals{
        uchar3{100, 150, 200},
        uchar3{50, 75, 125}
    };
    
    constexpr std::array<uchar4, 2> expectedVals{
        uchar4{100, 150, 200, 255},  // Alpha = maxDepthValue<p8bit> = 255
        uchar4{50, 75, 125, 255}
    };
    
    testCases[testName] = 
        TestCaseBuilder<fk::AddOpaqueAlpha<uchar3, fk::ColorDepth::p8bit>>::build(testName, inputVals, expectedVals);
}

// Test ColorConversion operations
void testColorConversionOperations() {
    // Test BGR2BGRA conversion (adds alpha)
    const std::string testName1 = "Test_ColorConversion_BGR2BGRA";
    
    constexpr std::array<uchar3, 2> inputVals1{
        uchar3{100, 150, 200},
        uchar3{50, 75, 125}
    };
    
    constexpr std::array<uchar4, 2> expectedVals1{
        uchar4{100, 150, 200, 255},
        uchar4{50, 75, 125, 255}
    };
    
    testCases[testName1] = 
        TestCaseBuilder<fk::ColorConversion<fk::ColorConversionCodes::COLOR_BGR2BGRA, uchar3, uchar4>>::build(testName1, inputVals1, expectedVals1);
    
    // Test BGR2RGB conversion (channel reorder)  
    const std::string testName2 = "Test_ColorConversion_BGR2RGB";
    
    constexpr std::array<uchar3, 2> inputVals2{
        uchar3{100, 150, 200},  // BGR
        uchar3{50, 75, 125}     // BGR
    };
    
    constexpr std::array<uchar3, 2> expectedVals2{
        uchar3{200, 150, 100},  // RGB (channels 2,1,0)
        uchar3{125, 75, 50}     // RGB (channels 2,1,0)
    };
    
    testCases[testName2] = 
        TestCaseBuilder<fk::ColorConversion<fk::ColorConversionCodes::COLOR_BGR2RGB, uchar3, uchar3>>::build(testName2, inputVals2, expectedVals2);
}

START_ADDING_TESTS
// Test UYVY pixel format traits
testUYVYPixelFormatTraits();

// Test IsEven function
testIsEvenFunction();

// Test RGB2Gray conversion
testRGB2GrayConversion();

// Test AddOpaqueAlpha
testAddOpaqueAlpha();

// Test ColorConversion operations
testColorConversionOperations();
STOP_ADDING_TESTS

int launch() {
    RUN_ALL_TESTS
}