/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

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

#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/warping.h>

int launch() {
    constexpr auto readIOp = fk::PerThreadRead<fk::_2D, uchar3>::build(
        fk::RawPtr<fk::_2D, uchar3>{ nullptr, { 128, 128, 128 * sizeof(uchar3) }});
    constexpr auto warpIOp =
        fk::Warping<fk::Perspective>::build(fk::WarpingParameters<fk::Perspective>{});
    constexpr auto fusedIOp = readIOp.then(warpIOp);

    constexpr bool correct =
        std::is_same_v<std::decay_t<decltype(fusedIOp)>,
        fk::ReadBack<fk::Warping<fk::Perspective, fk::Read<fk::PerThreadRead<fk::_2D, uchar3>>>>>;

    static_assert(correct, "Unexpected type for fusedIOp");

    return 0;
}