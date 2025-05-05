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

#include <array>

/* This code does not compile with gcc 13.3.0 with:
*   - CUDA SDK 12.8.1 (nvcc 12.8.93)
*   - CUDA SDK 12.9.0 (nvcc 12.9.41)
*
* It compiles fine with nvcc 12.8.93 or 12.9.41 + MSVC 19.42.34438.0 (MSVC 2022) on Windows
*/

void test1() {
    // Remove the constexpr std::size_t NUM, and use 5 directly, and it will compile
    constexpr std::size_t NUM = 5;
    const std::array<int, NUM> d_imgs{ 1, 2, 3, 4, 5 };
}

void test2() {
    // Change the std::array size to something different from 5 and it will compile
    const std::array<int, 5> d_imgs2{ 6, 7, 8, 9, 10 };
}

int launch() {
    return 0;
}
