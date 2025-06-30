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

#ifndef FK_TEST_VALUE_TO_STRING_H
#define FK_TEST_VALUE_TO_STRING_H

#include <fused_kernel/core/utils/value_to_string.h>

template <typename T, size_t N>
constexpr inline size_t usable_unsigned_count(const std::array<unsigned long long, N>& arr) {
    size_t count = 0;
    for (; count < N; ++count) {
        if (arr[count] > static_cast<unsigned long long>(std::numeric_limits<T>::max()))
            break;
    }
    return count;
}

template <typename T, size_t N>
constexpr size_t usable_signed_count(const std::array<long long, N>& arr) {
    size_t count = 0;
    for (; count < N; ++count) {
        if (arr[count] < static_cast<long long>(std::numeric_limits<T>::min()))
            break;
    }
    return count;
}

template <typename T>
constexpr inline bool test_count_digits_integral_helper() {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    constexpr std::array<unsigned long long, 20> unsigned_values = {
        1ULL,
        10ULL,
        100ULL,
        1000ULL,
        10000ULL,
        100000ULL,
        1000000ULL,
        10000000ULL,
        100000000ULL,
        1000000000ULL,
        10000000000ULL,
        100000000000ULL,
        1000000000000ULL,
        10000000000000ULL,
        100000000000000ULL,
        1000000000000000ULL,
        10000000000000000ULL,
        100000000000000000ULL,
        1000000000000000000ULL,
        10000000000000000000ULL
    };
    constexpr std::array<long long, 20> signed_values = {
        -4LL,
        -15LL,
        -106LL,
        -1020LL,
        -10700LL,
        -100000LL,
        -1000700LL,
        -10090000LL,
        -100001000LL,
        -1000400500LL,
        -10000000000LL,
        -100000500000LL,
        -1000000000000LL,
        -10000000000000LL,
        -100000000000000LL,
        -1000000800000000LL,
        -10000000000000000LL,
        -100000001200000000LL,
        -1000000000000000000LL,
        -10000000000000000000LL
    };
    constexpr size_t N = usable_unsigned_count<T>(unsigned_values);
    constexpr size_t M = usable_signed_count<T>(signed_values);

    for (size_t i = 0; i < N; ++i) {
        const T value = static_cast<T>(unsigned_values[i]);
        const size_t digitsUnsigned = fk::count_digits_integral(static_cast<ulonglong>(value));
        if (digitsUnsigned != (i + 1)) {
            return false;
        }
    }

    if (std::is_signed_v<T>) {
        for (size_t i = 0; i < M; ++i) {
            const T valueSigned = static_cast<T>(signed_values[i]);
            const ulonglong valueSignedAbs = static_cast<ulonglong>(-static_cast<longlong>(valueSigned));
            const size_t digitsSigned = fk::count_digits_integral(valueSignedAbs);
            if (digitsSigned != (i + 1)) {
                return false;
            }
        }
    }

    return true;
}

constexpr inline bool test_count_digits_integral() {
    // Test count_digits_integral for different integral types
    constexpr bool test_integral1 = test_count_digits_integral_helper<uchar>();
    constexpr bool test_integral2 = test_count_digits_integral_helper<ushort>();
    constexpr bool test_integral3 = test_count_digits_integral_helper<uint>();
    constexpr bool test_integral4 = test_count_digits_integral_helper<ulong>();
    constexpr bool test_integral5 = test_count_digits_integral_helper<ulonglong>();

    constexpr bool test_integral6 = test_count_digits_integral_helper<char>();
    constexpr bool test_integral7 = test_count_digits_integral_helper<short>();
    constexpr bool test_integral8 = test_count_digits_integral_helper<int>();
    constexpr bool test_integral9 = test_count_digits_integral_helper<long>();
    constexpr bool test_integral10 = test_count_digits_integral_helper<long long>();

    constexpr bool test_result = fk::and_v<test_integral1, test_integral2, test_integral3,
        test_integral4, test_integral5, test_integral6, test_integral7, test_integral8, test_integral9, test_integral10>;
    return test_result;
}

template <typename T>
constexpr inline bool test_getDigitAt_helper() {
    static_assert(std::is_integral_v<T>, "T must be an integral type");

    // Test positive numbers
    constexpr T pos_value = static_cast<T>(12345);
    if constexpr (sizeof(T) >= sizeof(int)) {
        // Test basic digit extraction
        static_assert(fk::getDigitAt<0>(pos_value) == '1', "getDigitAt<0>(12345) should return '1' (leftmost digit)");
        static_assert(fk::getDigitAt<1>(pos_value) == '2', "getDigitAt<1>(12345) should return '2' (second digit)");
        static_assert(fk::getDigitAt<2>(pos_value) == '3', "getDigitAt<2>(12345) should return '3' (third digit)");
        static_assert(fk::getDigitAt<3>(pos_value) == '4', "getDigitAt<3>(12345) should return '4' (fourth digit)");
        static_assert(fk::getDigitAt<4>(pos_value) == '5', "getDigitAt<4>(12345) should return '5' (rightmost digit)");

        // Test out of bounds (should return '!')
        static_assert(fk::getDigitAt<5>(pos_value) == '!', "getDigitAt<5>(12345) should return '!' for out of bounds access");
        static_assert(fk::getDigitAt<10>(pos_value) == '!', "getDigitAt<10>(12345) should return '!' for out of bounds access");
    }

    // Test negative numbers (if signed type)
    if constexpr (std::is_signed_v<T> && sizeof(T) >= sizeof(int)) {
        constexpr T neg_value = static_cast<T>(-3456);
        static_assert(fk::getDigitAt<0>(neg_value) == '3', "getDigitAt<0>(-3456) should return '3' (abs value: 3456, leftmost digit)");
        static_assert(fk::getDigitAt<1>(neg_value) == '4', "getDigitAt<1>(-3456) should return '4' (abs value: 3456, second digit)");
        static_assert(fk::getDigitAt<2>(neg_value) == '5', "getDigitAt<2>(-3456) should return '5' (abs value: 3456, third digit)");
        static_assert(fk::getDigitAt<3>(neg_value) == '6', "getDigitAt<3>(-3456) should return '6' (abs value: 3456, rightmost digit)");
        static_assert(fk::getDigitAt<4>(neg_value) == '!', "getDigitAt<4>(-3456) should return '!' for out of bounds access");
    }

    // Test zero
    constexpr T zero_value = static_cast<T>(0);
    static_assert(fk::getDigitAt<0>(zero_value) == '0', "getDigitAt<0>(0) should return '0' for zero value");
    static_assert(fk::getDigitAt<1>(zero_value) == '!', "getDigitAt<1>(0) should return '!' for out of bounds access on single digit");

    // Test single digit
    constexpr T single_digit = static_cast<T>(7);
    static_assert(fk::getDigitAt<0>(single_digit) == '7', "getDigitAt<0>(7) should return '7' for single digit value");
    static_assert(fk::getDigitAt<1>(single_digit) == '!', "getDigitAt<1>(7) should return '!' for out of bounds access on single digit");

    
    if constexpr (std::is_same_v<T, unsigned char>) {
        constexpr T max_val = static_cast<T>(255);  // 3 digits
        static_assert(fk::getDigitAt<0>(max_val) == '2', "getDigitAt<0>(255) should return '2' (leftmost digit of 255)");
        static_assert(fk::getDigitAt<1>(max_val) == '5', "getDigitAt<1>(255) should return '5' (middle digit of 255)");
        static_assert(fk::getDigitAt<2>(max_val) == '5', "getDigitAt<2>(255) should return '5' (rightmost digit of 255)");
        static_assert(fk::getDigitAt<3>(max_val) == '!', "getDigitAt<3>(255) should return '!' for out of bounds access");
    } else if constexpr (std::is_same_v<T, unsigned short>) {
        constexpr T max_val = static_cast<T>(65535);  // 5 digits
        static_assert(fk::getDigitAt<0>(max_val) == '6', "getDigitAt<0>(65535) should return '6' (leftmost digit of 65535)");
        static_assert(fk::getDigitAt<4>(max_val) == '5', "getDigitAt<4>(65535) should return '5' (rightmost digit of 65535)");
        static_assert(fk::getDigitAt<5>(max_val) == '!', "getDigitAt<5>(65535) should return '!' for out of bounds access");
    } else if constexpr (std::is_same_v<T, char>) {
        constexpr T max_val = static_cast<T>(127);  // 3 digits
        static_assert(fk::getDigitAt<0>(max_val) == '1', "getDigitAt<0>(127) should return '2' (leftmost digit of 127)");
        static_assert(fk::getDigitAt<1>(max_val) == '2', "getDigitAt<1>(127) should return '5' (middle digit of 127)");
        static_assert(fk::getDigitAt<2>(max_val) == '7', "getDigitAt<2>(127) should return '5' (rightmost digit of 127)");
        static_assert(fk::getDigitAt<3>(max_val) == '!', "getDigitAt<3>(127) should return '!' for out of bounds access");
        
        constexpr char neg_char = -45;  // abs: 45, 2 digits
        static_assert(fk::getDigitAt<0>(neg_char) == '4', "getDigitAt<0>(-45) should return '4' for negative char");
        static_assert(fk::getDigitAt<1>(neg_char) == '5', "getDigitAt<1>(-45) should return '5' for negative char");
        static_assert(fk::getDigitAt<2>(neg_char) == '!', "getDigitAt<2>(-45) should return '!' for out of bounds on negative char");
    } else if constexpr (std::is_same_v<T, short>) {
        // Test positive short values
        constexpr short pos_short = 1234;  // 4 digits
        static_assert(fk::getDigitAt<0>(pos_short) == '1', "getDigitAt<0>(1234) should return '1' for short type");
        static_assert(fk::getDigitAt<1>(pos_short) == '2', "getDigitAt<1>(1234) should return '2' for short type");
        static_assert(fk::getDigitAt<2>(pos_short) == '3', "getDigitAt<2>(1234) should return '3' for short type");
        static_assert(fk::getDigitAt<3>(pos_short) == '4', "getDigitAt<3>(1234) should return '4' for short type");
        static_assert(fk::getDigitAt<4>(pos_short) == '!', "getDigitAt<4>(1234) should return '!' for out of bounds on short");

        // Test negative short values
        constexpr short neg_short = -567;  // abs: 567, 3 digits
        static_assert(fk::getDigitAt<0>(neg_short) == '5', "getDigitAt<0>(-567) should return '5' for negative short");
        static_assert(fk::getDigitAt<1>(neg_short) == '6', "getDigitAt<1>(-567) should return '6' for negative short");
        static_assert(fk::getDigitAt<2>(neg_short) == '7', "getDigitAt<2>(-567) should return '7' for negative short");
        static_assert(fk::getDigitAt<3>(neg_short) == '!', "getDigitAt<3>(-567) should return '!' for out of bounds on negative short");

        // Test short max value (32767)
        constexpr short max_short = 32767;  // 5 digits
        static_assert(fk::getDigitAt<0>(max_short) == '3', "getDigitAt<0>(32767) should return '3' for short max");
        static_assert(fk::getDigitAt<1>(max_short) == '2', "getDigitAt<1>(32767) should return '2' for short max");
        static_assert(fk::getDigitAt<2>(max_short) == '7', "getDigitAt<2>(32767) should return '7' for short max");
        static_assert(fk::getDigitAt<3>(max_short) == '6', "getDigitAt<3>(32767) should return '6' for short max");
        static_assert(fk::getDigitAt<4>(max_short) == '7', "getDigitAt<4>(32767) should return '7' for short max");
        static_assert(fk::getDigitAt<5>(max_short) == '!', "getDigitAt<5>(32767) should return '!' for out of bounds on short max");

        // Test short min value (-32768)
        constexpr short min_short = -32768;  // abs: 32768, 5 digits
        static_assert(fk::getDigitAt<0>(min_short) == '3', "getDigitAt<0>(-32768) should return '3' for short min");
        static_assert(fk::getDigitAt<1>(min_short) == '2', "getDigitAt<1>(-32768) should return '2' for short min");
        static_assert(fk::getDigitAt<2>(min_short) == '7', "getDigitAt<2>(-32768) should return '7' for short min");
        static_assert(fk::getDigitAt<3>(min_short) == '6', "getDigitAt<3>(-32768) should return '6' for short min");
        static_assert(fk::getDigitAt<4>(min_short) == '8', "getDigitAt<4>(-32768) should return '8' for short min");
        static_assert(fk::getDigitAt<5>(min_short) == '!', "getDigitAt<5>(-32768) should return '!' for out of bounds on short min");
    }
    return true;
}

template <typename ListOfTypes, size_t... Idx>
constexpr inline bool test_getDigitAt_helper(const std::index_sequence<Idx...>&) {
    return ((test_getDigitAt_helper<fk::TypeAt_t<Idx, ListOfTypes>>()) && ...);
}

constexpr inline bool test_getDigitAt() {
    /*using ArithmeticTypes = fk::Remove
    test_getDigitAt_helper<fk::StandardTypes>(std::make_index_sequence<fk::StandardTypes::size>{});*/
    // Test different integral types
    test_getDigitAt_helper<uchar>();
    test_getDigitAt_helper<ushort>();
    test_getDigitAt_helper<uint>();
    test_getDigitAt_helper<ulong>();
    test_getDigitAt_helper<ulonglong>();

    test_getDigitAt_helper<char>();
    test_getDigitAt_helper<short>();
    test_getDigitAt_helper<int>();
    test_getDigitAt_helper<long>();
    test_getDigitAt_helper<longlong>();

    return true;
}

int launch() {
    static_assert(test_count_digits_integral(), "count_digits_integral failed");
    static_assert(test_getDigitAt(), "getDigitAt failed");

    constexpr float value{ 123456.045f };
    constexpr size_t stringThing = fk::count_fractional_digits(value);

    constexpr char myDigit = fk::getDigitAt<0>(45);

    constexpr bool correct = myDigit == '4';
    
    return 0;
}

#endif // !FK_TEST_VALUE_TO_STRING_H
