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

    // Helper: count digits for the fractional part until exact representation or max precision
    template <typename T>
    constexpr inline size_t count_fractional_digits(const T& value) {
        static_assert(std::is_floating_point<T>::value, "T must be a floating-point type");
        constexpr size_t max_digits = std::numeric_limits<T>::max_digits10;
        size_t digits = 0;
        const T abs_value = value < 0 ? -value : value;
        T frac = abs_value - static_cast<ulonglong>(abs_value);

        while (digits < max_digits && frac > T(0)) {
            frac *= T(10);
            ulonglong digit = static_cast<ulonglong>(frac);
            frac -= static_cast<T>(digit);
            ++digits;
            // Break if we've reached machine precision
            if (frac < std::numeric_limits<T>::epsilon()) break;
        }
        return digits;
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

    // Integral to array
    template <typename T, size_t N>
    constexpr inline std::array<char, N> integral_to_array_impl(T value) {
        static_assert(std::is_integral<T>::value, "T must be an integral type");
        std::array<char, N> arr{};
        size_t i = N;
        bool negative = false;
        ulonglong v;

        if constexpr (std::is_signed<T>::value) {
            if (value < 0) {
                negative = true;
                // Handle overflow for most negative value
                v = static_cast<ulonglong>(-(value + 1)) + 1;
            } else {
                v = static_cast<ulonglong>(value);
            }
        } else {
            v = static_cast<ulonglong>(value);
        }

        if (v == 0) {
            arr[--i] = '0';
        } else {
            while (v && i > 0) {
                arr[--i] = static_cast<char>('0' + (v % 10));
                v /= 10;
            }
        }
        if (negative && i > 0) {
            arr[--i] = '-';
        }
        return arr;
    }

    // Floating-point to array (auto precision)
    template <typename T, size_t N>
    constexpr inline std::array<char, N> float_to_array_impl_auto(const T& value) {
        std::array<char, N> arr = {};
        size_t pos = 0;

        // Handle special cases
        if (value != value) { // NaN check
            // For C++17 literal, we'll use a large finite value representation
            arr[pos++] = '0';
            arr[pos++] = '.';
            arr[pos++] = '0';
            if constexpr (std::is_same_v<T, float>) {
                arr[pos++] = 'f';
            }
            return arr;
        }

        T abs_value = value;
        if (value < 0) {
            arr[pos++] = '-';
            abs_value = -value;
        }

        ulonglong int_part = static_cast<ulonglong>(abs_value);

        // Write integer part
        auto int_digits = count_digits_integral(int_part);
        auto int_arr = integral_to_array_impl<ulonglong, 20>(int_part);
        for (size_t i = 20 - int_digits; i < 20; ++i) {
            arr[pos++] = int_arr[i];
        }

        // Always add decimal point for floating-point literals
        arr[pos++] = '.';

        // Write fractional part
        T frac = abs_value - static_cast<T>(int_part);
        size_t frac_digits = count_fractional_digits(abs_value);

        if (frac_digits == 0) {
            arr[pos++] = '0';
        } else {
            for (size_t i = 0; i < frac_digits; ++i) {
                frac *= T(10);
                ulonglong digit = static_cast<ulonglong>(frac);
                arr[pos++] = static_cast<char>('0' + digit);
                frac -= static_cast<T>(digit);
            }
        }

        // Add 'f' suffix for float literals
        if constexpr (std::is_same_v<T, float>) {
            arr[pos++] = 'f';
        }

        return arr;
    }

    // Main dispatcher
    template <typename T>
    constexpr inline auto value_to_array(const T& value) {
        static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");

        if constexpr (std::is_integral<T>::value) {
            // Handle char types specially for C++17 literals
            if constexpr (std::is_same_v<T, char>) {
                // Convert char to int for valid C++17 literal representation
                using IntType = int;
                IntType int_value = static_cast<IntType>(value);
                size_t n = (int_value < 0 ? 1 : 0) + count_digits_integral(int_value < 0 ? 
                    static_cast<ulonglong>(-static_cast<longlong>(int_value)) : static_cast<ulonglong>(int_value));
                return integral_to_array_impl<IntType, (n == 0 ? 1 : n)>(int_value);
            } else if constexpr (std::is_same_v<T, unsigned char>) {
                // Convert unsigned char to unsigned int
                using UIntType = unsigned int;
                UIntType uint_value = static_cast<UIntType>(value);
                size_t n = count_digits_integral(static_cast<ulonglong>(uint_value));
                return integral_to_array_impl<UIntType, (n == 0 ? 1 : n)>(uint_value);
            } else {
                size_t n = (value < 0 ? 1 : 0) + count_digits_integral(value < 0 ? -static_cast<ulonglong>(value) : static_cast<ulonglong>(value));
                return integral_to_array_impl<T, (n == 0 ? 1 : n)>(value);
            }
        } else if constexpr (std::is_floating_point<T>::value) {
            // Count integer and fractional digits
            const T abs_value = value < 0 ? -value : value;
            const size_t int_digits = count_digits_integral(static_cast<ulonglong>(abs_value));
            const size_t frac_digits = count_fractional_digits(abs_value);

            // Calculate total size: sign + int_digits + '.' + frac_digits + suffix
            size_t total = (value < 0 ? 1 : 0) + int_digits + 1 + (frac_digits == 0 ? 1 : frac_digits);
            if constexpr (std::is_same_v<T, float>) {
                total += 1; // for 'f' suffix
            }

            return float_to_array_impl_auto<T, total>(value);
        } else {
            static_assert(std::is_arithmetic<T>::value, "Unsupported type");
        }
    }

    template <typename T, typename = void>
    struct HasToString : std::false_type {};

    template <typename T>
    struct HasToString<T, ::std::void_t<decltype(::std::declval<T>().toString())>> : std::true_type {};

    template <typename T>
    constexpr bool hasToString_v = HasToString<T>::value;

    template <typename T>
    FK_HOST_INLINE auto toStringLiteral(const T& value) {
        static_assert(std::is_fundamental_v<T>, "The type must be a fundamental type.");

        std::ostringstream oss;
        if constexpr (std::is_same_v<T, char>) {
            // Convert char to int for valid C++17 literal representation
            return std::to_string(static_cast<int>(value));
        } else if constexpr (std::is_same_v<T, uchar>) {
            // Convert unsigned char to unsigned int for valid C++17 literal representation
            oss << static_cast<unsigned int>(value);
        } else if constexpr (std::is_same_v<T, float>) {
            // Handle float with 'f' suffix
            oss << std::setprecision(std::numeric_limits<float>::digits10 + 1)
                << std::scientific << value << "f";
        } else if constexpr (std::is_same_v<T, double>) {
            // Handle double without suffix
            oss << std::setprecision(std::numeric_limits<double>::digits10 + 1)
                << std::scientific << value;
        } else if constexpr (std::is_integral_v<T>) {
            // Handle integral types directly
            oss << value;
        } else {
            static_assert(std::is_fundamental_v<T>, "Unsupported type for C++17 literal generation.");
        }

        return oss.str();
    }

    template <size_t... Idx, typename T>
    FK_HOST_CNST auto serializeField_helper(const std::index_sequence<Idx...>&, const T& field) {
        return String("{ ") + ((toStringLiteral(field[Idx]) + String(", "))...) + String(" }");
    }

    template <typename T>
    FK_HOST_CNST auto serializeField(const T& field) {
        if constexpr (HasToString<T>::value) {
            // Recursively serialize nested structs
            return field.toString();
        } else if constexpr (std::is_array_v<T>) {
            return serializeField_helper(std::make_index_sequence<std::extent_v<T>>{}, field);
        } else if constexpr (IsCudaVector<T>::value) {
            return serialize(cudaVectorToTuple(field));
        } else {
            static_assert(std::is_fundamental_v<T>, "Type T is not fundamental.");
            return toStringLiteral(field);
        }
    }

    template <size_t... Idx, typename... Args>
    FK_HOST_CNST auto serialize_helper(const std::index_sequence<Idx...>&, const Tuple<Args...>& tuple) {
        return String("{ ") + (serializeField(fk::get<Idx>(tuple)) + ... + String(", ")) + String(" }");
    }

    template <typename... Args>
    FK_HOST_CNST auto serialize(const Tuple<Args...>&tuple) {
        return serialize_helper(std::make_index_sequence<Tuple<Args...>::size>{}, tuple);
    }
    
} // namespace fk
#endif // !defined(NVRTC_COMPILER)

#endif // FK_VALUE_TO_STRING_H
