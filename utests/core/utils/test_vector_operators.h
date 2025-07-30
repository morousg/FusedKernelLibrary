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

#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <type_traits>
#include <iostream>
#include <cassert>

// Helper function to test equality for floating point numbers
template <typename T>
bool approxEqual(T a, T b, T epsilon = static_cast<T>(1e-6)) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::abs(a - b) < epsilon;
    } else {
        return a == b;
    }
}

// Test comparison operators return correct types
template <typename VecType, typename BoolVecType>
void testComparisonOperatorTypes() {
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
template <typename VecType, typename BoolVecType>
void testScalarComparisonOperatorTypes() {
    static_assert(fk::validCUDAVec<VecType>, "Must be a valid CUDA vector type");
    
    using BaseType = fk::VBase<VecType>;
    VecType a = fk::make_set<VecType>(static_cast<BaseType>(1));
    BaseType scalar = static_cast<BaseType>(2);
    
    // Test vector-to-scalar comparisons
    auto eq_result = a == scalar;
    auto ne_result = a != scalar;
    auto gt_result = a > scalar;
    auto lt_result = a < scalar;
    auto ge_result = a >= scalar;
    auto le_result = a <= scalar;
    
    static_assert(std::is_same_v<decltype(eq_result), BoolVecType>, "== operator should return bool vector");
    static_assert(std::is_same_v<decltype(ne_result), BoolVecType>, "!= operator should return bool vector");
    static_assert(std::is_same_v<decltype(gt_result), BoolVecType>, "> operator should return bool vector");
    static_assert(std::is_same_v<decltype(lt_result), BoolVecType>, "< operator should return bool vector");
    static_assert(std::is_same_v<decltype(ge_result), BoolVecType>, ">= operator should return bool vector");
    static_assert(std::is_same_v<decltype(le_result), BoolVecType>, "<= operator should return bool vector");
    
    // Test scalar-to-vector comparisons (should be symmetric)
    auto seq_result = scalar == a;
    auto sne_result = scalar != a;
    auto sgt_result = scalar > a;
    auto slt_result = scalar < a;
    auto sge_result = scalar >= a;
    auto sle_result = scalar <= a;
    
    static_assert(std::is_same_v<decltype(seq_result), BoolVecType>, "== operator should return bool vector");
    static_assert(std::is_same_v<decltype(sne_result), BoolVecType>, "!= operator should return bool vector");
    static_assert(std::is_same_v<decltype(sgt_result), BoolVecType>, "> operator should return bool vector");
    static_assert(std::is_same_v<decltype(slt_result), BoolVecType>, "< operator should return bool vector");
    static_assert(std::is_same_v<decltype(sge_result), BoolVecType>, ">= operator should return bool vector");
    static_assert(std::is_same_v<decltype(sle_result), BoolVecType>, "<= operator should return bool vector");
}

// Test arithmetic operators return correct types
template <typename VecType>
void testArithmeticOperatorTypes() {
    static_assert(fk::validCUDAVec<VecType>, "Must be a valid CUDA vector type");
    
    using BaseType = fk::VBase<VecType>;
    VecType a = fk::make_set<VecType>(static_cast<BaseType>(2));
    VecType b = fk::make_set<VecType>(static_cast<BaseType>(1));
    BaseType scalar = static_cast<BaseType>(1);
    
    // Test that operators exist and compile
    [[maybe_unused]] auto add_result = a + b;
    [[maybe_unused]] auto sub_result = a - b;
    [[maybe_unused]] auto mul_result = a * b;
    [[maybe_unused]] auto div_result = a / b;
    
    // Test scalar operations
    [[maybe_unused]] auto add_scalar_result = a + scalar;
    [[maybe_unused]] auto sub_scalar_result = a - scalar;
    [[maybe_unused]] auto mul_scalar_result = a * scalar;
    [[maybe_unused]] auto div_scalar_result = a / scalar;
    
    // Test scalar on left side
    [[maybe_unused]] auto scalar_add_result = scalar + a;
    [[maybe_unused]] auto scalar_sub_result = scalar - a;
    [[maybe_unused]] auto scalar_mul_result = scalar * a;
    [[maybe_unused]] auto scalar_div_result = scalar / a;
}

// Test unary operators
template <typename VecType>
void testUnaryOperators() {
    static_assert(fk::validCUDAVec<VecType>, "Must be a valid CUDA vector type");
    
    using BaseType = fk::VBase<VecType>;
    VecType a = fk::make_set<VecType>(static_cast<BaseType>(1));
    
    // Test unary minus for signed types
    if constexpr (std::is_signed_v<fk::VBase<VecType>>) {
        [[maybe_unused]] auto neg_result = -a;
    }
    
    // Test logical not
    [[maybe_unused]] auto not_result = !a;
    
    // Test bitwise not for integral types
    if constexpr (std::is_integral_v<fk::VBase<VecType>>) {
        [[maybe_unused]] auto bnot_result = ~a;
    }
}

// Test compound assignment operators
template <typename VecType>
void testCompoundAssignmentOperators() {
    static_assert(fk::validCUDAVec<VecType>, "Must be a valid CUDA vector type");
    
    using BaseType = fk::VBase<VecType>;
    VecType a = fk::make_set<VecType>(static_cast<BaseType>(4));
    VecType b = fk::make_set<VecType>(static_cast<BaseType>(2));
    BaseType scalar = static_cast<BaseType>(1);
    
    // Test compound assignment with vectors
    a += b;
    a -= b;
    a *= b;
    a /= b;
    
    // Test compound assignment with scalars
    a += scalar;
    a -= scalar;
    a *= scalar;
    a /= scalar;
}

// Test bitwise operators for integral types
template <typename VecType>
void testBitwiseOperators() {
    static_assert(fk::validCUDAVec<VecType>, "Must be a valid CUDA vector type");
    
    if constexpr (std::is_integral_v<fk::VBase<VecType>>) {
        using BaseType = fk::VBase<VecType>;
        VecType a = fk::make_set<VecType>(static_cast<BaseType>(5));
        VecType b = fk::make_set<VecType>(static_cast<BaseType>(3));
        BaseType scalar = static_cast<BaseType>(1);
        
        // Test bitwise operations
        [[maybe_unused]] auto and_result = a & b;
        [[maybe_unused]] auto or_result = a | b;
        [[maybe_unused]] auto xor_result = a ^ b;
        
        // Test scalar bitwise operations
        [[maybe_unused]] auto and_scalar_result = a & scalar;
        [[maybe_unused]] auto or_scalar_result = a | scalar;
        [[maybe_unused]] auto xor_scalar_result = a ^ scalar;
        
        // Test scalar on left side
        [[maybe_unused]] auto scalar_and_result = scalar & a;
        [[maybe_unused]] auto scalar_or_result = scalar | a;
        [[maybe_unused]] auto scalar_xor_result = scalar ^ a;
    }
}

// Test actual computation for float2 to ensure operators work correctly
void testFloat2ComputationCorrectness() {
    float2 a = {3.0f, 4.0f};
    float2 b = {1.0f, 2.0f};
    
    // Test arithmetic
    auto add_result = a + b;
    assert(approxEqual(add_result.x, 4.0f) && approxEqual(add_result.y, 6.0f));
    
    auto sub_result = a - b;
    assert(approxEqual(sub_result.x, 2.0f) && approxEqual(sub_result.y, 2.0f));
    
    auto mul_result = a * b;
    assert(approxEqual(mul_result.x, 3.0f) && approxEqual(mul_result.y, 8.0f));
    
    auto div_result = a / b;
    assert(approxEqual(div_result.x, 3.0f) && approxEqual(div_result.y, 2.0f));
    
    // Test scalar arithmetic
    auto scalar_add = a + 1.0f;
    assert(approxEqual(scalar_add.x, 4.0f) && approxEqual(scalar_add.y, 5.0f));
    
    auto scalar_mul = a * 2.0f;
    assert(approxEqual(scalar_mul.x, 6.0f) && approxEqual(scalar_mul.y, 8.0f));
    
    // Test comparisons
    auto eq_result = a == b;
    assert(!eq_result.x && !eq_result.y);
    
    auto gt_result = a > b;
    assert(gt_result.x && gt_result.y);
    
    auto scalar_gt = a > 2.0f;
    assert(scalar_gt.x && scalar_gt.y);
    
    auto scalar_lt = a < 2.0f;
    assert(!scalar_lt.x && !scalar_lt.y);
}

// Test actual computation for int4 to ensure operators work correctly
void testInt4ComputationCorrectness() {
    int4 a = {10, 20, 30, 40};
    int4 b = {5, 10, 15, 20};
    
    // Test arithmetic
    auto add_result = a + b;
    assert(add_result.x == 15 && add_result.y == 30 && add_result.z == 45 && add_result.w == 60);
    
    auto sub_result = a - b;
    assert(sub_result.x == 5 && sub_result.y == 10 && sub_result.z == 15 && sub_result.w == 20);
    
    // Test bitwise operations
    auto and_result = a & b;
    assert(and_result.x == (10 & 5) && and_result.y == (20 & 10) && 
           and_result.z == (30 & 15) && and_result.w == (40 & 20));
    
    // Test comparisons
    auto gt_result = a > b;
    assert(gt_result.x && gt_result.y && gt_result.z && gt_result.w);
    
    auto eq_result = a == a;
    assert(eq_result.x && eq_result.y && eq_result.z && eq_result.w);
    
    // Test scalar operations
    auto scalar_add = a + 5;
    assert(scalar_add.x == 15 && scalar_add.y == 25 && scalar_add.z == 35 && scalar_add.w == 45);
    
    auto scalar_gt = a > 25;
    assert(!scalar_gt.x && !scalar_gt.y && scalar_gt.z && scalar_gt.w);
}

int launch() {
    // Test comparison operator types for all vector types
    testComparisonOperatorTypes<float1, bool1>();
    testComparisonOperatorTypes<float2, bool2>();
    testComparisonOperatorTypes<float3, bool3>();
    testComparisonOperatorTypes<float4, bool4>();
    
    testComparisonOperatorTypes<int1, bool1>();
    testComparisonOperatorTypes<int2, bool2>();
    testComparisonOperatorTypes<int3, bool3>();
    testComparisonOperatorTypes<int4, bool4>();
    
    testComparisonOperatorTypes<uchar1, bool1>();
    testComparisonOperatorTypes<uchar2, bool2>();
    testComparisonOperatorTypes<uchar3, bool3>();
    testComparisonOperatorTypes<uchar4, bool4>();
    
    testComparisonOperatorTypes<double1, bool1>();
    testComparisonOperatorTypes<double2, bool2>();
    testComparisonOperatorTypes<double3, bool3>();
    testComparisonOperatorTypes<double4, bool4>();
    
    testComparisonOperatorTypes<long1, bool1>();
    testComparisonOperatorTypes<long2, bool2>();
    testComparisonOperatorTypes<long3, bool3>();
    testComparisonOperatorTypes<long4, bool4>();
    
    testComparisonOperatorTypes<longlong1, bool1>();
    testComparisonOperatorTypes<longlong2, bool2>();
    testComparisonOperatorTypes<longlong3, bool3>();
    testComparisonOperatorTypes<longlong4, bool4>();
    
    // Test scalar comparison operator types
    testScalarComparisonOperatorTypes<float1, bool1>();
    testScalarComparisonOperatorTypes<float2, bool2>();
    testScalarComparisonOperatorTypes<float3, bool3>();
    testScalarComparisonOperatorTypes<float4, bool4>();
    
    testScalarComparisonOperatorTypes<int1, bool1>();
    testScalarComparisonOperatorTypes<int2, bool2>();
    testScalarComparisonOperatorTypes<int3, bool3>();
    testScalarComparisonOperatorTypes<int4, bool4>();
    
    testScalarComparisonOperatorTypes<uchar1, bool1>();
    testScalarComparisonOperatorTypes<uchar2, bool2>();
    testScalarComparisonOperatorTypes<uchar3, bool3>();
    testScalarComparisonOperatorTypes<uchar4, bool4>();
    
    testScalarComparisonOperatorTypes<long1, bool1>();
    testScalarComparisonOperatorTypes<long2, bool2>();
    testScalarComparisonOperatorTypes<long3, bool3>();
    testScalarComparisonOperatorTypes<long4, bool4>();
    
    // Test arithmetic operators
    testArithmeticOperatorTypes<float2>();
    testArithmeticOperatorTypes<int4>();
    testArithmeticOperatorTypes<double3>();
    testArithmeticOperatorTypes<uchar1>();
    
    // Test unary operators
    testUnaryOperators<float2>();
    testUnaryOperators<int4>();
    testUnaryOperators<uchar3>();
    
    // Test compound assignment operators
    testCompoundAssignmentOperators<float2>();
    testCompoundAssignmentOperators<int4>();
    testCompoundAssignmentOperators<double1>();
    
    // Test bitwise operators
    testBitwiseOperators<int2>();
    testBitwiseOperators<uchar4>();
    testBitwiseOperators<short3>();
    
    // Test actual computation correctness
    testFloat2ComputationCorrectness();
    testInt4ComputationCorrectness();
    
    std::cout << "All vector operator tests passed!" << std::endl;
    return 0;
}

#endif