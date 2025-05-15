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

#ifdef __CUDACC__  // Only if we're compiling with nvcc
  // Define a composite version number for easier comparison.
  #define NVCC_VERSION (__CUDACC_VER_MAJOR__ * 10000 + \
                        __CUDACC_VER_MINOR__ * 100 +    \
                        __CUDACC_VER_BUILD__)

  // Define composite numbers for the particular versions you're interested in.
  #define NVCC_12_8_93 (12 * 10000 + 8 * 100 + 93)
  #define NVCC_12_9_41 (12 * 10000 + 9 * 100 + 41)

  #if NVCC_VERSION == NVCC_12_8_93
    #pragma message("nvcc version 12.8.93 detected! Test will be skipped.")
    // Code specific to nvcc version 12.8.93 can go here.
  #elif NVCC_VERSION == NVCC_12_9_41
    #pragma message("nvcc version 12.9.41 detected! Test will be skipped.")
    // Code specific to nvcc version 12.9.41 can go here.
  #else
    #pragma message("Compiling with nvcc version that should compile.")
void test1() {
    // Remove the constexpr std::size_t NUM, and use 5 directly, and it will compile
    constexpr std::size_t NUM = 5;
    [[maybe_unused]] const std::array<int, NUM> d_imgs{ 1, 2, 3, 4, 5 };
}
void test2() {
    // Change the std::array size to something different from 5 and it will compile
    [[maybe_unused]] const std::array<int, 5> d_imgs2{ 6, 7, 8, 9, 10 };
}
  #endif
#endif

int launch() {
    return 0;
}
