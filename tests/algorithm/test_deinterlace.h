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

#include <tests/main.h>

#include <fused_kernel/algorithms/image_processing/interpolation.h>
#include <fused_kernel/core/execution_model/memory_operations.h>

using namespace fk;

int launch() {
    constexpr RawPtr<ND::_2D, uchar3> input{ nullptr, {128, 64, 128*sizeof(uchar3)}};
    constexpr Size src_size(128, 64);
    constexpr auto readIOp = PerThreadRead<ND::_2D, uchar3>::build(input);
    using ReadIOp = decltype(readIOp);

    // Test DEINTERLACE_BLEND
    constexpr InterpolationParameters<InterpolationType::DEINTERLACE_BLEND> blendParams{ src_size };
    constexpr auto blendOp = Interpolate<InterpolationType::DEINTERLACE_BLEND, ReadIOp>::build(blendParams, readIOp);
    static_assert(isTernaryType<std::decay_t<decltype(blendOp)>>, "Deinterlace blend should be a ternary operation");

    // Test DEINTERLACE_INTER_LINEAR with field_select = 0 (even rows, interpolate odd)
    constexpr InterpolationParameters<InterpolationType::DEINTERLACE_INTER_LINEAR> interLinearParams0{ src_size, 0 };
    constexpr auto interLinearOp0 = Interpolate<InterpolationType::DEINTERLACE_INTER_LINEAR, ReadIOp>::build(interLinearParams0, readIOp);
    static_assert(isTernaryType<std::decay_t<decltype(interLinearOp0)>>, "Deinterlace inter linear should be a ternary operation");

    // Test DEINTERLACE_INTER_LINEAR with field_select = 1 (odd rows, interpolate even)
    constexpr InterpolationParameters<InterpolationType::DEINTERLACE_INTER_LINEAR> interLinearParams1{ src_size, 1 };
    constexpr auto interLinearOp1 = Interpolate<InterpolationType::DEINTERLACE_INTER_LINEAR, ReadIOp>::build(interLinearParams1, readIOp);
    static_assert(isTernaryType<std::decay_t<decltype(interLinearOp1)>>, "Deinterlace inter linear should be a ternary operation");

    return 0;
}