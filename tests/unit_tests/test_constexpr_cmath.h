/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_TEST_CONSTEXPR_CMATH_H
#define FK_TEST_CONSTEXPR_CMATH_H

#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>
#include <limits>
#include <cmath>

// Test isnan function
template <typename T>
constexpr bool test_isnan() {
    static_assert(std::is_floating_point_v<T>, "isnan test only for floating point types");
    
    // Test with normal values
    static_assert(!cxp::isnan(T(0.0)), "0.0 should not be NaN");
    static_assert(!cxp::isnan(T(1.0)), "1.0 should not be NaN");
    static_assert(!cxp::isnan(T(-1.0)), "-1.0 should not be NaN");
    static_assert(!cxp::isnan(std::numeric_limits<T>::max()), "max value should not be NaN");
    static_assert(!cxp::isnan(std::numeric_limits<T>::min()), "min value should not be NaN");
    static_assert(!cxp::isnan(std::numeric_limits<T>::lowest()), "lowest value should not be NaN");
    static_assert(!cxp::isnan(std::numeric_limits<T>::infinity()), "infinity should not be NaN");
    static_assert(!cxp::isnan(-std::numeric_limits<T>::infinity()), "-infinity should not be NaN");
    
    // Test with NaN
    static_assert(cxp::isnan(std::numeric_limits<T>::quiet_NaN()), "quiet_NaN should be NaN");
    static_assert(cxp::isnan(std::numeric_limits<T>::signaling_NaN()), "signaling_NaN should be NaN");
    
    return true;
}

// Test isinf function
template <typename T>
constexpr bool test_isinf() {
    static_assert(std::is_floating_point_v<T>, "isinf test only for floating point types");
    
    // Test with normal values
    static_assert(!cxp::isinf(T(0.0)), "0.0 should not be infinite");
    static_assert(!cxp::isinf(T(1.0)), "1.0 should not be infinite");
    static_assert(!cxp::isinf(T(-1.0)), "-1.0 should not be infinite");
    static_assert(!cxp::isinf(T(1000.0)), "1000.0 should not be infinite");
    static_assert(!cxp::isinf(T(-1000.0)), "-1000.0 should not be infinite");
    static_assert(!cxp::isinf(std::numeric_limits<T>::quiet_NaN()), "NaN should not be infinite");
    
    // Test with infinity
    static_assert(cxp::isinf(std::numeric_limits<T>::infinity()), "infinity should be infinite");
    static_assert(cxp::isinf(-std::numeric_limits<T>::infinity()), "-infinity should be infinite");
    
    return true;
}

// Test cmp_equal function
constexpr bool test_cmp_equal() {
    // Same type comparisons
    static_assert(cxp::cmp_equal(5, 5), "5 == 5 should be true");
    static_assert(!cxp::cmp_equal(5, 4), "5 == 4 should be false");
    static_assert(cxp::cmp_equal(5.0, 5.0), "5.0 == 5.0 should be true");
    static_assert(!cxp::cmp_equal(5.0, 4.0), "5.0 == 4.0 should be false");
    
    // Mixed signed/unsigned comparisons
    static_assert(cxp::cmp_equal(5, 5u), "5 == 5u should be true");
    static_assert(!cxp::cmp_equal(-1, 5u), "-1 == 5u should be false");
    static_assert(!cxp::cmp_equal(5u, -1), "5u == -1 should be false");
    static_assert(cxp::cmp_equal(0, 0u), "0 == 0u should be true");
    
    // Mixed integer/floating point comparisons
    static_assert(cxp::cmp_equal(5, 5.0), "5 == 5.0 should be true");
    static_assert(cxp::cmp_equal(5.0, 5), "5.0 == 5 should be true");
    static_assert(!cxp::cmp_equal(5, 5.1), "5 == 5.1 should be false");
    
    return true;
}

// Test cmp_not_equal function
constexpr bool test_cmp_not_equal() {
    static_assert(!cxp::cmp_not_equal(5, 5), "5 != 5 should be false");
    static_assert(cxp::cmp_not_equal(5, 4), "5 != 4 should be true");
    static_assert(cxp::cmp_not_equal(-1, 5u), "-1 != 5u should be true");
    static_assert(!cxp::cmp_not_equal(5, 5.0), "5 != 5.0 should be false");
    
    return true;
}

// Test cmp_less function
constexpr bool test_cmp_less() {
    // Same type comparisons
    static_assert(cxp::cmp_less(4, 5), "4 < 5 should be true");
    static_assert(!cxp::cmp_less(5, 4), "5 < 4 should be false");
    static_assert(!cxp::cmp_less(5, 5), "5 < 5 should be false");
    
    // Mixed signed/unsigned comparisons
    static_assert(cxp::cmp_less(-1, 5u), "-1 < 5u should be true");
    static_assert(!cxp::cmp_less(5u, -1), "5u < -1 should be false");
    static_assert(cxp::cmp_less(4u, 5), "4u < 5 should be true");
    static_assert(!cxp::cmp_less(5u, 4), "5u < 4 should be false");
    
    // Mixed integer/floating point comparisons
    static_assert(cxp::cmp_less(4, 5.0), "4 < 5.0 should be true");
    static_assert(cxp::cmp_less(4.0, 5), "4.0 < 5 should be true");
    static_assert(!cxp::cmp_less(5.0, 4), "5.0 < 4 should be false");
    
    return true;
}

// Test cmp_greater function
constexpr bool test_cmp_greater() {
    static_assert(cxp::cmp_greater(5, 4), "5 > 4 should be true");
    static_assert(!cxp::cmp_greater(4, 5), "4 > 5 should be false");
    static_assert(!cxp::cmp_greater(5, 5), "5 > 5 should be false");
    static_assert(cxp::cmp_greater(5u, -1), "5u > -1 should be true");
    static_assert(!cxp::cmp_greater(-1, 5u), "-1 > 5u should be false");
    
    return true;
}

// Test cmp_less_equal function
constexpr bool test_cmp_less_equal() {
    static_assert(cxp::cmp_less_equal(4, 5), "4 <= 5 should be true");
    static_assert(cxp::cmp_less_equal(5, 5), "5 <= 5 should be true");
    static_assert(!cxp::cmp_less_equal(5, 4), "5 <= 4 should be false");
    static_assert(cxp::cmp_less_equal(-1, 5u), "-1 <= 5u should be true");
    static_assert(cxp::cmp_less_equal(0, 0u), "0 <= 0u should be true");
    
    return true;
}

// Test cmp_greater_equal function
constexpr bool test_cmp_greater_equal() {
    static_assert(cxp::cmp_greater_equal(5, 4), "5 >= 4 should be true");
    static_assert(cxp::cmp_greater_equal(5, 5), "5 >= 5 should be true");
    static_assert(!cxp::cmp_greater_equal(4, 5), "4 >= 5 should be false");
    static_assert(cxp::cmp_greater_equal(5u, -1), "5u >= -1 should be true");
    static_assert(cxp::cmp_greater_equal(0u, 0), "0u >= 0 should be true");
    
    return true;
}

// Test round function
template <typename T>
constexpr bool test_round() {
    static_assert(std::is_floating_point_v<T>, "round test only for floating point types");
    
    static_assert(cxp::round(T(1.4)) == T(1.0), "round(1.4) should be 1.0");
    static_assert(cxp::round(T(1.5)) == T(2.0), "round(1.5) should be 2.0");
    static_assert(cxp::round(T(1.6)) == T(2.0), "round(1.6) should be 2.0");
    static_assert(cxp::round(T(-1.4)) == T(-1.0), "round(-1.4) should be -1.0");
    static_assert(cxp::round(T(-1.5)) == T(-2.0), "round(-1.5) should be -2.0");
    static_assert(cxp::round(T(-1.6)) == T(-2.0), "round(-1.6) should be -2.0");
    static_assert(cxp::round(T(0.0)) == T(0.0), "round(0.0) should be 0.0");
    static_assert(cxp::round(T(2.0)) == T(2.0), "round(2.0) should be 2.0");
    
    // Test with special values
    static_assert(cxp::isnan(cxp::round(std::numeric_limits<T>::quiet_NaN())), "round(NaN) should be NaN");
    static_assert(cxp::isinf(cxp::round(std::numeric_limits<T>::infinity())), "round(inf) should be inf");
    static_assert(cxp::isinf(cxp::round(-std::numeric_limits<T>::infinity())), "round(-inf) should be -inf");
    
    return true;
}

// Test abs function
template <typename T>
constexpr bool test_abs() {
    static_assert(std::is_fundamental_v<T>, "abs test only for fundamental types");
    
    if constexpr (std::is_signed_v<T>) {
        static_assert(cxp::abs(T(5)) == T(5), "abs(5) should be 5");
        static_assert(cxp::abs(T(-5)) == T(5), "abs(-5) should be 5");
        static_assert(cxp::abs(T(0)) == T(0), "abs(0) should be 0");
        
        // Test edge case: most negative value
        static_assert(cxp::abs(cxp::minValue<T>) == cxp::maxValue<T>, 
                      "abs(min) should be max for signed types");
    } else {
        // Unsigned types
        static_assert(cxp::abs(T(5)) == T(5), "abs(5) should be 5 for unsigned");
        static_assert(cxp::abs(T(0)) == T(0), "abs(0) should be 0 for unsigned");
    }
    
    return true;
}

// Test max function
constexpr bool test_max() {
    static_assert(cxp::max(5) == 5, "max(5) should be 5");
    static_assert(cxp::max(3, 5) == 5, "max(3, 5) should be 5");
    static_assert(cxp::max(5, 3) == 5, "max(5, 3) should be 5");
    static_assert(cxp::max(1, 3, 5, 2) == 5, "max(1, 3, 5, 2) should be 5");
    static_assert(cxp::max(-1, -3, -5, -2) == -1, "max(-1, -3, -5, -2) should be -1");
    static_assert(cxp::max(1.0, 3.0, 5.0, 2.0) == 5.0, "max(1.0, 3.0, 5.0, 2.0) should be 5.0");
    
    return true;
}

// Test min function - Note: there's a bug in the implementation that we expect to catch
constexpr bool test_min() {
    static_assert(cxp::min(5) == 5, "min(5) should be 5");
    
    // Test with two arguments to see if the bug manifests
    // The bug in min_helper calls max_helper instead of min_helper recursively
    static_assert(cxp::min(3, 5) == 3, "min(3, 5) should be 3");
    static_assert(cxp::min(5, 3) == 3, "min(5, 3) should be 3");
    static_assert(cxp::min(1, 3, 5, 2) == 1, "min(1, 3, 5, 2) should be 1");
    static_assert(cxp::min(-1, -3, -5, -2) == -5, "min(-1, -3, -5, -2) should be -5");
    static_assert(cxp::min(1.0, 3.0, 5.0, 2.0) == 1.0, "min(1.0, 3.0, 5.0, 2.0) should be 1.0");
    
    return true;
}

// Runtime tests to complement compile-time tests
bool runtime_tests() {
    // Test isnan with runtime values
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    if (!cxp::isnan(nan_val)) return false;
    if (cxp::isnan(1.0f)) return false;
    
    // Test isinf with runtime values
    float inf_val = std::numeric_limits<float>::infinity();
    if (!cxp::isinf(inf_val)) return false;
    if (!cxp::isinf(-inf_val)) return false;
    if (cxp::isinf(1.0f)) return false;
    if (cxp::isinf(std::numeric_limits<float>::max())) return false;
    if (cxp::isinf(std::numeric_limits<float>::min())) return false;
    if (cxp::isinf(std::numeric_limits<float>::lowest())) return false;
    
    // Test comparison functions with runtime values
    if (!cxp::cmp_equal(5, 5)) return false;
    if (cxp::cmp_equal(5, 4)) return false;
    if (!cxp::cmp_less(4, 5)) return false;
    if (cxp::cmp_less(5, 4)) return false;
    
    // Test mixed type comparisons
    if (!cxp::cmp_equal(5u, 5)) return false;
    if (cxp::cmp_equal(5u, -1)) return false;
    if (!cxp::cmp_less(-1, 5u)) return false;
    if (cxp::cmp_less(5u, -1)) return false;
    
    // Test round with runtime values
    if (cxp::round(1.4f) != 1.0f) return false;
    if (cxp::round(1.6f) != 2.0f) return false;
    if (cxp::round(-1.4f) != -1.0f) return false;
    if (cxp::round(-1.6f) != -2.0f) return false;
    if (cxp::round(0.0f) != 0.0f) return false;
    
    // Test round with special values
    if (!cxp::isnan(cxp::round(nan_val))) return false;
    if (!cxp::isinf(cxp::round(inf_val))) return false;
    if (!cxp::isinf(cxp::round(-inf_val))) return false;
    
    // Test abs with runtime values
    if (cxp::abs(-5) != 5) return false;
    if (cxp::abs(5) != 5) return false;
    if (cxp::abs(0) != 0) return false;
    if (cxp::abs(-1.5f) != 1.5f) return false;
    if (cxp::abs(1.5f) != 1.5f) return false;
    
    // Test abs with min value edge case
    if (cxp::abs(cxp::minValue<int>) != cxp::maxValue<int>) return false;
    
    // Test max with runtime values
    if (cxp::max(3, 5, 1) != 5) return false;
    if (cxp::max(-1, -3, -5) != -1) return false;
    
    // Test min with runtime values
    if (cxp::min(3, 5, 1) != 1) return false;
    if (cxp::min(-1, -3, -5) != -5) return false;
    
    return true;
}

int launch() {
    // Compile-time tests
    static_assert(test_isnan<float>(), "isnan test failed for float");
    static_assert(test_isnan<double>(), "isnan test failed for double");
    
    static_assert(test_isinf<float>(), "isinf test failed for float");
    static_assert(test_isinf<double>(), "isinf test failed for double");
    
    static_assert(test_cmp_equal(), "cmp_equal test failed");
    static_assert(test_cmp_not_equal(), "cmp_not_equal test failed");
    static_assert(test_cmp_less(), "cmp_less test failed");
    static_assert(test_cmp_greater(), "cmp_greater test failed");
    static_assert(test_cmp_less_equal(), "cmp_less_equal test failed");
    static_assert(test_cmp_greater_equal(), "cmp_greater_equal test failed");
    
    static_assert(test_round<float>(), "round test failed for float");
    static_assert(test_round<double>(), "round test failed for double");
    
    static_assert(test_abs<int>(), "abs test failed for int");
    static_assert(test_abs<float>(), "abs test failed for float");
    static_assert(test_abs<double>(), "abs test failed for double");
    static_assert(test_abs<unsigned int>(), "abs test failed for unsigned int");
    
    static_assert(test_max(), "max test failed");
    static_assert(test_min(), "min test failed");
    
    // Runtime tests
    if (!runtime_tests()) {
        return -1;
    }
    
    return 0;
}

#endif