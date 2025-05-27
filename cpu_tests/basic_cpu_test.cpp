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

#include "../include/fused_kernel/fused_kernel.h"
#include "../include/fused_kernel/core/data/ptr_nd.h"

using namespace fk;

int main() {

    Ptr2D<float> input(1024, 1024, 0, MemType::Host);
    Ptr2D<float> output(1024, 1024, 0, MemType::Host);

    executeOperations<ParArch::CPU, DPPType::Transform>(PerThreadRead<_2D, float>::build(input),
                                                        PerThreadWrite<_2D, float>::build(output));

    return 0;
}