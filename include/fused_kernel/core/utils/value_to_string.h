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
#include <sstream>
#include <iomanip>
#include <utility>
#include <type_traits>

namespace fk {
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

        template <typename... Args>
        FK_HOST_CNST auto serialize(const Tuple<Args...>&tuple) {
            return serialize_helper(std::make_index_sequence<Tuple<Args...>::size>{}, tuple);
        }

    } // namespace fk
#endif // !defined(NVRTC_COMPILER)

#endif // FK_VALUE_TO_STRING_H
