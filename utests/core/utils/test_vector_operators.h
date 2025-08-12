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

#ifndef FK_TEST_VECTOR_OPERATORS_H
#define FK_TEST_VECTOR_OPERATORS_H

#include <bitset>
#include <cassert>
#include <fused_kernel/algorithms/basic_ops/cuda_vector.h>
#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/core/utils/type_to_string.h>
#include <iostream>
#include <type_traits>

// Helper function to test equality for floating point numbers
bool approxEqualF(const float &a, const float &b) {
    constexpr float epsilon = 1e-5f; // Define a small epsilon value for floating point comparison
    return std::abs(a - b) < epsilon;
}

bool approxEqualD(const float &a, const double &b) {
    constexpr float epsilon = 1e-5f; // Define a small epsilon value for floating point comparison
    return std::abs(a - b) < epsilon;
}
// Test comparison operators return correct types
template <typename VecType, typename BoolVecType> void testComparisonOperatorTypes() {
    static_assert(fk::validCUDAVec<VecType>, "Must be a valid CUDA vector type");

    // Initialize with non-zero values to avoid division by zero issues
    using BaseType = fk::VBase<VecType>;
    VecType a = fk::make_set<VecType>(static_cast<BaseType>(1));
    VecType b = fk::make_set<VecType>(static_cast<BaseType>(2));

    // Test vector-to-vector comparisons
    auto eq_result = a == b;
    auto ne_result = a != b;
    auto gt_result = a > b;
    auto lt_result = a < b;
    auto ge_result = a >= b;
    auto le_result = a <= b;

    static_assert(std::is_same_v<decltype(eq_result), BoolVecType>, "== operator should return bool vector");
    static_assert(std::is_same_v<decltype(ne_result), BoolVecType>, "!= operator should return bool vector");
    static_assert(std::is_same_v<decltype(gt_result), BoolVecType>, "> operator should return bool vector");
    static_assert(std::is_same_v<decltype(lt_result), BoolVecType>, "< operator should return bool vector");
    static_assert(std::is_same_v<decltype(ge_result), BoolVecType>, ">= operator should return bool vector");
    static_assert(std::is_same_v<decltype(le_result), BoolVecType>, "<= operator should return bool vector");

    // Test logical operators
    auto and_result = a && b;
    auto or_result = a || b;
    static_assert(std::is_same_v<decltype(and_result), BoolVecType>, "&& operator should return bool vector");
    static_assert(std::is_same_v<decltype(or_result), BoolVecType>, "|| operator should return bool vector");
}

// Test scalar comparison operators return correct types
template <typename VecType, typename BoolVecType> void testScalarComparisonOperatorTypes() {
    static_assert(fk::validCUDAVec<VecType>, "Must be a valid CUDA vector type");

    using BaseType = fk::VBase<VecType>;
    constexpr VecType a = fk::make_set<VecType>(static_cast<BaseType>(1));
    constexpr BaseType scalar = static_cast<BaseType>(2);

    // Test vector-to-scalar comparisons
    constexpr auto eq_result = a == scalar;
    constexpr auto ne_result = a != scalar;
    constexpr auto gt_result = a > scalar;
    constexpr auto lt_result = a < scalar;
    constexpr auto ge_result = a >= scalar;
    constexpr auto le_result = a <= scalar;

    static_assert(std::is_same_v<decltype(eq_result), const BoolVecType>, "== operator should return bool vector");
    static_assert(std::is_same_v<decltype(ne_result), const BoolVecType>, "!= operator should return bool vector");
    static_assert(std::is_same_v<decltype(gt_result), const BoolVecType>, "> operator should return bool vector");
    static_assert(std::is_same_v<decltype(lt_result), const BoolVecType>, "< operator should return bool vector");
    static_assert(std::is_same_v<decltype(ge_result), const BoolVecType>, ">= operator should return bool vector");
    static_assert(std::is_same_v<decltype(le_result), const BoolVecType>, "<= operator should return bool vector");

    // Test scalar-to-vector comparisons (should be symmetric)
    constexpr auto seq_result = scalar == a;
    constexpr auto sne_result = scalar != a;
    constexpr auto sgt_result = scalar > a;
    constexpr auto slt_result = scalar < a;
    constexpr auto sge_result = scalar >= a;
    constexpr auto sle_result = scalar <= a;
    
    static_assert(std::is_same_v<decltype(seq_result), const BoolVecType>, "== operator should return bool vector");
    static_assert(std::is_same_v<decltype(sne_result), const BoolVecType>, "!= operator should return bool vector");
    static_assert(std::is_same_v<decltype(sgt_result), const BoolVecType>, "> operator should return bool vector");
    static_assert(std::is_same_v<decltype(slt_result), const BoolVecType>, "< operator should return bool vector");
    static_assert(std::is_same_v<decltype(sge_result), const BoolVecType>, ">= operator should return bool vector");
    static_assert(std::is_same_v<decltype(sle_result), const BoolVecType>, "<= operator should return bool vector");
}

// Test arithmetic operators return correct types
template <typename VecType, typename ReturnType> bool testArithmeticOperatorTypes() {
    bool result = true;
    static_assert(fk::validCUDAVec<VecType>, "Must be a valid CUDA vector type");

    using BaseType = fk::VBase<VecType>;
    constexpr VecType a = fk::make_set<VecType>(static_cast<BaseType>(15));
    constexpr VecType b = fk::make_set<VecType>(static_cast<BaseType>(5));
    constexpr BaseType scalar = static_cast<BaseType>(1);

    // Test that operators exist and compile
    constexpr auto add_result = a + b;
    static_assert(std::is_same_v<decltype(add_result), ReturnType>, "type should not change after operation: ");
    if (!fk::vecAnd(add_result == fk::make_set<VecType>(static_cast<BaseType>(20)))) {
        assert(false && "arithmetic assignment with vector failed");
        result = false;
    }
    constexpr auto sub_result = a - b;
    static_assert(std::is_same_v<decltype(sub_result), ReturnType>, "type should not change after operation");
    if (!fk::vecAnd(sub_result == fk::make_set<VecType>(static_cast<BaseType>(10)))) {
        assert(false && "arithmetic assignment with vector failed");
        result = false;
    }
    constexpr auto mul_result = a * b;
    static_assert(std::is_same_v<decltype(mul_result), ReturnType>, "type should not change after operation");
    if (!fk::vecAnd(mul_result == fk::make_set<VecType>(static_cast<BaseType>(75)))) {
        assert(false && "arithmetic assignment with vector failed");
        result = false;
    }
    constexpr auto div_result = a / b;
    static_assert(std::is_same_v<decltype(mul_result), ReturnType>, "type should not change after operation");
    if (!fk::vecAnd(div_result == fk::make_set<VecType>(static_cast<BaseType>(3)))) {
        assert(false && "arithmetic assignment with vector failed");
        result = false;
    }
    // Test scalar operations
    constexpr auto add_scalar_result = a + scalar;
    static_assert(std::is_same_v<decltype(add_scalar_result), ReturnType>, "type should not change after operation");
    if (!fk::vecAnd(add_scalar_result == fk::make_set<VecType>(static_cast<BaseType>(16)))) {
        assert(false && "scalar assignment with vector failed");
        result = false;
    }
    constexpr auto sub_scalar_result = a - scalar;
    static_assert(std::is_same_v<decltype(sub_scalar_result), ReturnType>, "type should not change after operation");
    static_assert(std::is_same_v<decltype(add_scalar_result), ReturnType>, "type should not change after operation");
    if (!fk::vecAnd(sub_scalar_result == fk::make_set<VecType>(static_cast<BaseType>(14)))) {
        assert(false && "scalar assignment with vector failed");
        result = false;
    }
    constexpr auto mul_scalar_result = a * scalar;
    static_assert(std::is_same_v<decltype(add_scalar_result), ReturnType>, "type should not change after operation");
    if (!fk::vecAnd(mul_scalar_result == fk::make_set<VecType>(static_cast<BaseType>(16)))) {
        assert(false && "scalar assignment with vector failed");
        result = false;
    }

    static_assert(std::is_same_v<decltype(mul_scalar_result), ReturnType>, "type should not change after operation");
    constexpr auto div_scalar_result = a / scalar;
    static_assert(std::is_same_v<decltype(div_scalar_result), ReturnType>, "type should not change after operation");
    if (!fk::vecAnd(div_scalar_result == fk::make_set<VecType>(static_cast<BaseType>(16)))) {
        assert(false && "scalar assignment with vector failed");
        result = false;
    }
    // Test scalar on left side
    constexpr auto scalar_add_result = scalar + a;
    static_assert(std::is_same_v<decltype(scalar_add_result), ReturnType>, "type should not change after operation");
    if (!fk::vecAnd(scalar_add_result == fk::make_set<VecType>(static_cast<BaseType>(16)))) {
        assert(false && "scalar assignment with vector failed");
        result = false;
    }

    constexpr auto scalar_sub_result = scalar - a;
    static_assert(std::is_same_v<decltype(scalar_sub_result), ReturnType>, "type should not change after operation");
    if (!fk::vecAnd(scalar_sub_result == fk::make_set<VecType>(static_cast<BaseType>(14)))) {
        assert(false && "scalar assignment with vector failed");
        result = false;
    }
    constexpr auto scalar_mul_result = scalar * a;
    static_assert(std::is_same_v<decltype(scalar_mul_result), ReturnType>, "type should not change after operation");
    if (!fk::vecAnd(scalar_mul_result == fk::make_set<VecType>(static_cast<BaseType>(16)))) {
        assert(false && "scalar assignment with vector failed");
        result = false;
    }

    constexpr auto scalar_div_result = scalar / a;
    static_assert(std::is_same_v<decltype(scalar_div_result), ReturnType>, "type should not change after operation");
    if (!fk::vecAnd(scalar_div_result == fk::make_set<VecType>(static_cast<BaseType>(3)))) {
        assert(false && "arithmetic assignment with vector failed");
        result = false;
    }
}

// Test unary operators
// TODO review this:
template <typename VecType> bool testUnaryOperators() {
    /* static_assert(fk::validCUDAVec<VecType>, "Must be a valid CUDA vector type");

    using BaseType = fk::VBase<VecType>;
    VecType a = fk::make_set<VecType>(static_cast<BaseType>(1));

    // Test unary minus for signed types
    if constexpr (std::is_signed_v<fk::VBase<VecType>>) {
        auto neg_result = -a;
    }

    // Test logical not
    [[maybe_unused]] auto not_result = !a;

    // Test bitwise not for integral types
    if constexpr (std::is_integral_v<fk::VBase<VecType>>) {
        [[maybe_unused]] auto bnot_result = ~a;
    }*/
    return true;
}

// Test compound assignment operators
template <typename VecType> bool testCompoundAssignmentOperators() {
    bool result = true;
    static_assert(fk::validCUDAVec<VecType>, "Must be a valid CUDA vector type");

    using BaseType = fk::VBase<VecType>;
    VecType a = fk::make_set<VecType>(static_cast<BaseType>(4));
    VecType b = fk::make_set<VecType>(static_cast<BaseType>(2));
    BaseType scalar = static_cast<BaseType>(3);

    // Test compound assignment with vectors
    a += b;

    if (!fk::vecAnd(a == fk::make_set<VecType>(static_cast<BaseType>(6)))) {
        assert(false && "Compound assignment with vector failed");
        result = false;
    }
    a -= b;
    if (!fk::vecAnd(a == fk::make_set<VecType>(static_cast<BaseType>(4)))) {
        assert(false && "Compound assignment with vector failed");
        result = false;
    }
    a *= b;
    if (!fk::vecAnd(a == fk::make_set<VecType>(static_cast<BaseType>(8)))) {
        assert(false && "Compound assignment with vector failed");
        result = false;
    }
    a /= b;
    if (!fk::vecAnd(a == fk::make_set<VecType>(static_cast<BaseType>(4)))) {
        assert(false && "Compound assignment with vector failed");
        result = false;
    }
    // Test compound assignment with scalars
    a += scalar;
    if (!fk::vecAnd(a == fk::make_set<VecType>(static_cast<BaseType>(7)))) {
        assert(false && "Compound assignment with vector failed");
        result = false;
    }
    a -= scalar;
    if (!fk::vecAnd(a == fk::make_set<VecType>(static_cast<BaseType>(4)))) {
        assert(false && "Compound assignment with vector failed");
        result = false;
    }
    a *= scalar;
    if (!fk::vecAnd(a == fk::make_set<VecType>(static_cast<BaseType>(12)))) {
        assert(false && "Compound assignment with vector failed");
        result = false;
    }
    a /= scalar;
    if (!fk::vecAnd(a == fk::make_set<VecType>(static_cast<BaseType>(4)))) {
        assert(false && "Compound assignment with vector failed");
        result = false;
    }
    return result;
}

// Test bitwise operators for integral types
template <typename VecType> void testBitwiseOperators() {
    static_assert(fk::validCUDAVec<VecType>, "Must be a valid CUDA vector type");

    if constexpr (std::is_integral_v<fk::VBase<VecType>>) {
        using BaseType = fk::VBase<VecType>;
        constexpr VecType a = fk::make_set<VecType>(static_cast<BaseType>(5));
        constexpr VecType b = fk::make_set<VecType>(static_cast<BaseType>(3));
        constexpr BaseType scalar = static_cast<BaseType>(1);

        // Test bitwise operations
        constexpr auto and_result = a & b;
        static_assert(fk::vecAnd(and_result == fk::make_set<VecType>(static_cast<BaseType>(1))),
                      "Compound assignment with vector failed");

        constexpr auto or_result = a | b;
        static_assert(fk::vecAnd(or_result == fk::make_set<VecType>(static_cast<BaseType>(7))),
                      "Compound assignment with vector failed");

        constexpr auto xor_result = a ^ b;
        static_assert(fk::vecAnd(xor_result == fk::make_set<VecType>(static_cast<BaseType>(6))),
                      "Compound assignment with vector failed");
        // Test scalar bitwise operations
        constexpr auto and_scalar_result = a & scalar;
        static_assert(fk::vecAnd(and_scalar_result == fk::make_set<VecType>(static_cast<BaseType>(1))),
                      "Compound assignment with vector failed");

        constexpr auto or_scalar_result = a | scalar;
        static_assert(fk::vecAnd(or_scalar_result == fk::make_set<VecType>(static_cast<BaseType>(5))),
                      "Compound assignment with vector failed");

        constexpr auto xor_scalar_result = a ^ scalar;
        static_assert(fk::vecAnd(xor_scalar_result == fk::make_set<VecType>(static_cast<BaseType>(4))),
                      "Compound assignment with vector failed");

        // Test scalar on left side using constexpr and static_assert
        constexpr auto scalar_and_result = scalar & a;
        static_assert(fk::vecAnd(scalar_and_result == fk::make_set<VecType>(static_cast<BaseType>(1))),
                      "Compound assignment with vector failed");

        constexpr auto scalar_or_result = scalar | a;
        static_assert(fk::vecAnd(scalar_or_result == fk::make_set<VecType>(static_cast<BaseType>(5))),
                      "Compound assignment with vector failed");

        constexpr auto scalar_xor_result = scalar ^ a;
        static_assert(fk::vecAnd(scalar_xor_result == fk::make_set<VecType>(static_cast<BaseType>(4))),
                      "Compound assignment with vector failed");
    }
}

// Test actual computation for float2 to ensure operators work correctly
bool testFloat2ComputationCorrectness() {
    constexpr float2 a = {3.0f, 4.0f};
    constexpr float2 b = {1.0f, 2.0f};
    bool res = true;
    // Test arithmetic
    constexpr auto add_result = a + b;

    if (!approxEqualF(add_result.x, 4.0f) && approxEqualF(add_result.y, 6.0f)) {
        std::cerr << "Float2 addition test failed: expected (4.0, 6.0), got (" << add_result.x << ", " << add_result.y
                  << ")" << std::endl;
        res = false;
    }

    auto sub_result = a - b;
    if (!(approxEqualF(sub_result.x, 2.0f) && approxEqualF(sub_result.y, 2.0f))) {
        std::cerr << "Float2 subtraction test failed: expected (2.0, 2.0), got (" << sub_result.x << ", "
                  << sub_result.y << ")" << std::endl;
        res = false;
    }

    auto mul_result = a * b;
    if (!(approxEqualF(mul_result.x, 3.0f) && approxEqualF(mul_result.y, 8.0f))) {
        std::cerr << "Float2 multiplication test failed: expected (3.0, 8.0), got (" << mul_result.x << ", "
                  << mul_result.y << ")" << std::endl;
        res = false;
    }

    auto div_result = a / b;
    if (!(approxEqualF(div_result.x, 3.0f) && approxEqualF(div_result.y, 2.0f))) {
        std::cerr << "Float2 division test failed: expected (3.0, 2.0), got (" << div_result.x << ", " << div_result.y
                  << ")" << std::endl;
        res = false;
    }

    // Test scalar arithmetic
    auto scalar_add = a + 1.0f;
    if (!(approxEqualF(scalar_add.x, 4.0f) && approxEqualF(scalar_add.y, 5.0f))) {
        std::cerr << "Float2 scalar addition test failed: expected (4.0, 5.0), got (" << scalar_add.x << ", "
                  << scalar_add.y << ")" << std::endl;
        res = false;
    }

    auto scalar_mul = a * 2.0f;
    if (!(approxEqualF(scalar_mul.x, 6.0f) && approxEqualF(scalar_mul.y, 8.0f))) {
        std::cerr << "Float2 scalar multiplication test failed: expected (6.0, 8.0), got (" << scalar_mul.x << ", "
                  << scalar_mul.y << ")" << std::endl;
        res = false;
    }

    // Test comparisons
    auto eq_result = a == b;
    if (!(!eq_result.x && !eq_result.y)) {
        std::cerr << "Float2 equality test failed: expected (false, false), got (" << eq_result.x << ", " << eq_result.y
                  << ")" << std::endl;
        res = false;
    }

    auto gt_result = a > b;
    if (!(gt_result.x && gt_result.y)) {
        std::cerr << "Float2 greater than test failed: expected (true, true), got (" << gt_result.x << ", "
                  << gt_result.y << ")" << std::endl;
        res = false;
    }
    auto scalar_gt = a > 2.0f;
    if (!(scalar_gt.x && scalar_gt.y)) {
        std::cerr << "Float2 scalar greater than test failed: expected (true, true), got (" << scalar_gt.x << ", "
                  << scalar_gt.y << ")" << std::endl;
        res = false;
    }

    auto scalar_lt = a < 2.0f;
    if (!(!scalar_lt.x && !scalar_lt.y)) {
        std::cerr << "Float2 scalar less than test failed: expected (false, false), got (" << scalar_lt.x << ", "
                  << scalar_lt.y << ")" << std::endl;
        res = false;
    }
    return res;
}

// Test actual computation for float2 to ensure operators work correctly
bool testDouble2ComputationCorrectness() {
    constexpr double2 a = {3.0, 4.0};
    constexpr double2 b = {1.0, 2.0};
    bool res = true;
    // Test arithmetic
    constexpr auto add_result = a + b;

    if (!approxEqualD(add_result.x, 4.0) && approxEqualD(add_result.y, 6.0)) {
        std::cerr << "Double2 addition test failed: expected (4.0, 6.0), got (" << add_result.x << ", " << add_result.y
                  << ")" << std::endl;
        res = false;
    }

    auto sub_result = a - b;
    if (!(approxEqualD(sub_result.x, 2.0) && approxEqualD(sub_result.y, 2.0))) {
        std::cerr << "Double2 subtraction test failed: expected (2.0, 2.0), got (" << sub_result.x << ", "
                  << sub_result.y << ")" << std::endl;
        res = false;
    }

    auto mul_result = a * b;
    if (!(approxEqualD(mul_result.x, 3.0) && approxEqualD(mul_result.y, 8.0))) {
        std::cerr << "Double2 multiplication test failed: expected (3.0, 8.0), got (" << mul_result.x << ", "
                  << mul_result.y << ")" << std::endl;
        res = false;
    }

    auto div_result = a / b;
    if (!(approxEqualD(div_result.x, 3.0) && approxEqualD(div_result.y, 2.0))) {
        std::cerr << "Double2 division test failed: expected (3.0, 2.0), got (" << div_result.x << ", " << div_result.y
                  << ")" << std::endl;
        res = false;
    }

    // Test scalar arithmetic
    auto scalar_add = a + 1.0;
    if (!(approxEqualD(scalar_add.x, 4.0) && approxEqualD(scalar_add.y, 5.0))) {
        std::cerr << "Double2 scalar addition test failed: expected (4.0, 5.0), got (" << scalar_add.x << ", "
                  << scalar_add.y << ")" << std::endl;
        res = false;
    }

    auto scalar_mul = a * 2.0;
    if (!(approxEqualD(scalar_mul.x, 6.0) && approxEqualD(scalar_mul.y, 8.0))) {
        std::cerr << "Double2 scalar multiplication test failed: expected (6.0, 8.0), got (" << scalar_mul.x << ", "
                  << scalar_mul.y << ")" << std::endl;
        res = false;
    }

    // Test comparisons
    auto eq_result = a == b;
    if (!(!eq_result.x && !eq_result.y)) {
        std::cerr << "Double2 equality test failed: expected (false, false), got (" << eq_result.x << ", "
                  << eq_result.y << ")" << std::endl;
        res = false;
    }

    auto gt_result = a > b;
    if (!(gt_result.x && gt_result.y)) {
        std::cerr << "Double2 greater than test failed: expected (true, true), got (" << gt_result.x << ", "
                  << gt_result.y << ")" << std::endl;
        res = false;
    }
    auto scalar_gt = a > 2.0;
    if (!(scalar_gt.x && scalar_gt.y)) {
        std::cerr << "Double2 scalar greater than test failed: expected (true, true), got (" << scalar_gt.x << ", "
                  << scalar_gt.y << ")" << std::endl;
        res = false;
    }

    auto scalar_lt = a < 2.0;
    if (!(!scalar_lt.x && !scalar_lt.y)) {
        std::cerr << "Double2 scalar less than test failed: expected (false, false), got (" << scalar_lt.x << ", "
                  << scalar_lt.y << ")" << std::endl;
        res = false;
    }
    return res;
}

template <typename BaseInput, typename BaseOutput> void addOneTestAllChannelsOpTypes() {

    // Vector of 1
    using Input1 = typename fk::VectorType<BaseInput, 1>::type_v;
    using Output1 = typename fk::VectorType<BaseOutput, 1>::type_v;
    testComparisonOperatorTypes<Input1, bool1>();
    testScalarComparisonOperatorTypes<Input1, bool1>();
    testUnaryOperators<Input1>();
    testBitwiseOperators<Input1>();

    testCompoundAssignmentOperators<Input1>();
    // Vector of 2
    using Input2 = fk::VectorType_t<BaseInput, 2>;
    using Output2 = fk::VectorType_t<BaseOutput, 2>;
    testComparisonOperatorTypes<Input2, bool2>();
    testScalarComparisonOperatorTypes<Input2, bool2>();
    testUnaryOperators<Input2>();
    testCompoundAssignmentOperators<Input2>();
    testBitwiseOperators<Input2>();

    // Vector of 3
    using Input3 = fk::VectorType_t<BaseInput, 3>;
    using Output3 = fk::VectorType_t<BaseOutput, 3>;
    testComparisonOperatorTypes<Input3, bool3>();
    testScalarComparisonOperatorTypes<Input3, bool3>();
    testUnaryOperators<Input3>();
    testCompoundAssignmentOperators<Input3>();
    testBitwiseOperators<Input3>();

    // Vector of 4
    using Input4 = fk::VectorType_t<BaseInput, 4>;
    using Output4 = fk::VectorType_t<BaseOutput, 4>;
    testComparisonOperatorTypes<Input4, bool4>();
    testScalarComparisonOperatorTypes<Input4, bool4>();
    testUnaryOperators<Input4>();
    testCompoundAssignmentOperators<Input4>();
    testBitwiseOperators<Input4>();
}

template <typename TypeList_, typename Type, size_t... Idx>
void addAllTestsFor_helper_op_types(const std::index_sequence<Idx...> &) {
    static_assert(fk::validCUDAVec<Type> || std::is_fundamental_v<Type>,
                  "Type must be either a cuda vector or a fundamental type.");
    static_assert(fk::isTypeList<TypeList_>, "TypeList_ must be a valid TypeList.");
    // For each type in TypeList_, add tests with Type
    (addOneTestAllChannelsOpTypes<fk::TypeAt_t<Idx, TypeList_>, Type>(), ...);
}

template <typename TypeList_, size_t... Idx> void addAllTestsForOpTypes(const std::index_sequence<Idx...> &) {
    // For each type in TypeList_, add tests with each type in TypeList_
    (addAllTestsFor_helper_op_types<TypeList_, fk::TypeAt_t<Idx, TypeList_>>(
         std::make_index_sequence<TypeList_::size>{}),
     ...);
}

int launch() {

    // Test boolean operators
    using Fundamental = fk::RemoveType_t<0, fk::RemoveType_t<0, fk::RemoveType_t<0, fk::StandardTypes>>>;
    addAllTestsForOpTypes<Fundamental>(std::make_index_sequence<Fundamental::size>());
 

 

    bool resF = testFloat2ComputationCorrectness();
    bool resD = testDouble2ComputationCorrectness();
    if (!resF || !resD  ) {
        std::cerr << "Vector operator tests failed!" << std::endl;
        return -1;
    }
    std::cout << "All vector operator tests passed!" << std::endl;

    return 0;
}

#endif