/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Hguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_STRING_H
#define FK_STRING_H

#if !defined(NVRTC_COMPILER)
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>

namespace fk {
    template <size_t N>
    class String {
    private:
        template <size_t... Idx1, size_t... Idx2, size_t M>
        constexpr inline String<sizeof...(Idx1) + sizeof...(Idx2) + 1>
        cat_helper(const std::index_sequence<Idx1...>&, const std::index_sequence<Idx2...>&,
                   const String<M>& other) const noexcept {
            return String<sizeof...(Idx1) + sizeof...(Idx2) + 1>({ data[Idx1]..., other.data[Idx2]... });
        }

        template <size_t... Idx>
        constexpr inline bool operator_equal_helper(const std::index_sequence<Idx...>&, const String<N>& other) const noexcept {
            return ((data[Idx] == other.data[Idx]) && ...);
        }

    public:
        std::array<char, N> data{};
        size_t length{ 0 };
        // Default constructor
        constexpr String() = default;

        // Constructor from a String literal
        constexpr String(const char(&str)[N]) {
            for (size_t i = 0; i < N - 1; ++i) {
                data[i] = str[i];
            }
            length = N - 1;
        }

        // Size of the String
        constexpr size_t size() const noexcept {
            return length;
        }

        // Maximum capacity
        constexpr size_t capacity() const noexcept {
            return N;
        }

        // Access character at index (no bounds checking)
        constexpr char operator[](size_t index) const noexcept {
            return data[index];
        }

        // Access character at index (with bounds checking)
        constexpr char at(size_t index) const {
            if (index >= length) {
                throw std::out_of_range("Index out of range");
            }
            return data[index];
        }

        // Concatenate two strings
        template <size_t M>
        constexpr String<(N + M) - 1> operator+(const String<M>& other) const noexcept {
            return cat_helper(std::make_index_sequence<N - 1>{}, std::make_index_sequence<M - 1>{}, other);
        }

        template <size_t M>
        constexpr auto operator+(const char(&other)[M]) const noexcept {
            return *this + String<M>(other);
        }

        // Substring
        constexpr String<N> substr(size_t start, size_t count) const {
            if (start >= length || start + count > length) {
                throw std::out_of_range("Substring out of range");
            }
            String<N> result;
            for (size_t i = 0; i < count; ++i) {
                result.data[i] = data[start + i];
            }
            result.length = count;
            return result;
        }

        // Get raw data
        constexpr const char* c_str() const noexcept {
            return data.data();
        }

        std::string str() const noexcept {
            return data.data();
        }

        std::string operator()() const noexcept {
            return str();
        }

        template <size_t M>
        constexpr bool operator==(const String<M>& other) const noexcept {
            if constexpr (N != M) {
                return false;
            } else {
                return operator_equal_helper(std::make_index_sequence<N - 1>{}, other);
            }
        }

        template <size_t M>
        constexpr bool operator==(const char(&other)[M]) const noexcept {
            return this == String(other);
        }

        // Less than operator
        template <size_t M>
        constexpr bool operator<(const String<M>& other) const noexcept {
            for (size_t i = 0; i < std::min(length, other.length); ++i) {
                if (data[i] < other.data[i]) {
                    return true;
                } else if (data[i] > other.data[i]) {
                    return false;
                }
            }
            return length < other.length;
        }
    };
} // namespace fk

template <size_t N>
std::ostream& operator<<(std::ostream& os, const fk::String<N>& str) {
    return os << str.c_str();
}

#endif

#endif // FK_STRING_H