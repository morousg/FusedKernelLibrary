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

#ifndef FK_VALUE_TO_STRING_H
#define FK_VALUE_TO_STRING_H

#if !defined(NVRTC_COMPILER)
#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/data/string.h>
#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>
#include <sstream>
#include <iomanip>
#include <utility>
#include <type_traits>
#include <array>
#include <limits>

namespace fk {
    // Count digits for integral types
    constexpr inline size_t count_digits_integral(const ulonglong& v) {
        size_t count = 0;
        ulonglong temp = v;
        do {
            ++count;
            temp /= 10;
        } while (temp);
        return count;
    }

    template <size_t Idx, typename T>
    constexpr inline char getDigitAt(const T& value) {
        static_assert(std::is_integral<T>::value, "T must be an integral type");

        constexpr std::array<ulonglong, 20> powers10{ 1ULL, 10ULL, 100ULL, 1000ULL, 10000ULL, 100000ULL,
        1000000ULL, 10000000ULL, 100000000ULL, 1000000000ULL,
        10000000000ULL, 100000000000ULL, 1000000000000ULL,
        10000000000000ULL, 100000000000000ULL, 1000000000000000ULL,
        10000000000000000ULL, 100000000000000000ULL,
        1000000000000000000ULL, 10000000000000000000ULL };

        const bool isMinVal = value < 0 && value == std::numeric_limits<T>::lowest();

        const ulonglong absValue = static_cast<ulonglong>(isMinVal ? 
            static_cast<ulonglong>(-static_cast<longlong>(std::numeric_limits<T>::lowest())) : cxp::abs(value));
        const size_t totalDigits = count_digits_integral(absValue);
        const size_t powerIndex = totalDigits - Idx - 1;
        if (totalDigits <= Idx) {
            return '!';
        }
        const ulonglong divisor = powers10[powerIndex];
        if (value == 0) {
            return '0';
        } else {
            const ulonglong digit = (absValue / divisor) % 10;
            return static_cast<char>('0' + digit);
        }
    }

    template <typename T, typename = void>
    struct HasToString : std::false_type {};

    template <typename T>
    struct HasToString<T, ::std::void_t<decltype(::std::declval<T>().toString())>> : std::true_type {};

    template <typename T>
    constexpr bool hasToString_v = HasToString<T>::value;

    // TODO: we prioritize runtime compilation without serialization of runtime data.

} // namespace fk
#endif // !defined(NVRTC_COMPILER)

#endif // FK_VALUE_TO_STRING_H
