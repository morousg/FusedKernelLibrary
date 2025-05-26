/* Copyright 2024 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <array>

#include <fused_kernel/core/utils/vlimits.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>

#include <iostream>

template <size_t START_VALUE, size_t INCREMENT, std::size_t... Is>
constexpr std::array<size_t, sizeof...(Is)> generate_sequence(std::index_sequence<Is...>) {
    return std::array<size_t, sizeof...(Is)>{(START_VALUE + (INCREMENT * Is))...};
}

template <size_t START_VALUE, size_t INCREMENT, size_t NUM_ELEMS>
constexpr std::array<size_t, NUM_ELEMS> arrayIndexSecuence = generate_sequence<START_VALUE, INCREMENT>(std::make_index_sequence<NUM_ELEMS>{});

template <typename T>
inline bool compareAndCheck(const fk::Ptr2D<T>& firstResult, const fk::Ptr2D<T>& secondResult) {
    const bool sameDims = firstResult.dims().width == secondResult.dims().width && firstResult.dims().height == secondResult.dims().height;
    if (!sameDims) {
        std::cout << "Dimensions do not match: " << firstResult.dims().width << "x" << firstResult.dims().height << " vs " << secondResult.dims().width << "x" << secondResult.dims().height << std::endl;
        return false;
    }
    for (uint y = 0; y < firstResult.dims().height; ++y) {
        for (uint x = 0; x < firstResult.dims().width; ++x) {
            if (!fk::Equal<T>::exec(fk::make_tuple(firstResult.at(fk::Point(x, y)), secondResult.at(fk::Point(x, y))))) {
                std::cout << "Mismatch at (" << x << ", " << y << ") " << std::endl;
                return false;
            }
        }
    }
    return true;
}