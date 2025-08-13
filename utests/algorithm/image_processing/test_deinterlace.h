/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <fused_kernel/algorithms/image_processing/deinterlace.h>
#include <fused_kernel/core/data/size.h>

namespace fk {
namespace tests {

// Test that the deinterlace types are properly defined
constexpr bool test_deinterlace_types() {
    static_assert(static_cast<int>(DeinterlaceType::BLEND) == 0, "BLEND type should be 0");
    static_assert(static_cast<int>(DeinterlaceType::INTER_LINEAR) == 1, "INTER_LINEAR type should be 1");
    return true;
}

// Test that parameter structures can be instantiated
constexpr bool test_deinterlace_parameters() {
    constexpr Size test_size{64, 64};
    
    [[maybe_unused]] constexpr DeinterlaceParameters<DeinterlaceType::BLEND> blend_params{test_size};
    [[maybe_unused]] constexpr DeinterlaceParameters<DeinterlaceType::INTER_LINEAR> linear_params{test_size};
    
    return true;
}

// Test basic template instantiation
constexpr bool test_deinterlace_instantiation() {
    using BlendDeinterlace = Deinterlace<DeinterlaceType::BLEND, void>;
    using LinearDeinterlace = Deinterlace<DeinterlaceType::INTER_LINEAR, void>;
    
    static_assert(std::is_same_v<BlendDeinterlace, BlendDeinterlace>, "BlendDeinterlace type check");
    static_assert(std::is_same_v<LinearDeinterlace, LinearDeinterlace>, "LinearDeinterlace type check");
    
    return true;
}

} // namespace tests
} // namespace fk

// Main test launcher function (outside fk namespace)
int launch() {
    static_assert(fk::tests::test_deinterlace_types(), "Deinterlace types test failed");
    static_assert(fk::tests::test_deinterlace_parameters(), "Deinterlace parameters test failed");
    static_assert(fk::tests::test_deinterlace_instantiation(), "Deinterlace instantiation test failed");
    
    // Runtime tests (could be expanded later)
    const bool all_tests_passed = 
        fk::tests::test_deinterlace_types() &&
        fk::tests::test_deinterlace_parameters() &&
        fk::tests::test_deinterlace_instantiation();
    
    return all_tests_passed ? 0 : 1;
}