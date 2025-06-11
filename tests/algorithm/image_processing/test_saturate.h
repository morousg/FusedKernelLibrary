/* Copyright 2025 the Fused Kernel Project Developers

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "fused_kernel/algorithms/image_processing/saturate.h"
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/cuda_vector.h>
#include <fused_kernel/core/utils/type_to_string.h>
#include <fused_kernel/core/utils/vlimits.h>

#include <map>
#include <functional>
#include <iostream> 

using namespace fk;

// This helper simply forces one more round of macro expansion.
// Standard concatenation and stringification
#define CONCAT_INNER(a, b) a##b
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define STRINGIFY_INNER(x) #x
#define STRINGIFY(x) STRINGIFY_INNER(x)

// Removes a single set of parentheses: DEPAREN((a,b)) -> a,b
#define DEPAREN_IMPL(...) __VA_ARGS__
#define DEPAREN(x) DEPAREN_IMPL x

// =========================================================================
// 2. Variadic Argument Concatenation (the magic part)
//
// This takes multiple arguments (e.g., a, b, c) and creates a single
// token with underscores (a_b_c). It supports 1 to 8 identifiers.
// =========================================================================

// Macros to concatenate a specific number of arguments with underscores
#define CAT_WITH_UNDERSCORE_1(a) a
#define CAT_WITH_UNDERSCORE_2(a, b) a##_##b
#define CAT_WITH_UNDERSCORE_3(a, b, c) a##_##b##_##c
#define CAT_WITH_UNDERSCORE_4(a, b, c, d) a##_##b##_##c##_##d
#define CAT_WITH_UNDERSCORE_5(a, b, c, d, e) a##_##b##_##c##_##d##_##e
#define CAT_WITH_UNDERSCORE_6(a, b, c, d, e, f) a##_##b##_##c##_##d##_##e##_##f
#define CAT_WITH_UNDERSCORE_7(a, b, c, d, e, f, g) a##_##b##_##c##_##d##_##e##_##f##_##g
#define CAT_WITH_UNDERSCORE_8(a, b, c, d, e, f, g, h) a##_##b##_##c##_##d##_##e##_##f##_##g##_##h

// This helper simply forces one more round of macro expansion.
#define EXPAND(x) x

// The helper that selects the 9th argument from a list.
#define GET_9TH_ARG(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N

// The reverse sequence of numbers, now including 0 for the zero-argument case.
#define REVERSE_SEQ_8() 8, 7, 6, 5, 4, 3, 2, 1, 0

// The robust implementation of COUNT_VARARGS
// The extra _IMPL layer ensures __VA_ARGS__ is expanded correctly before counting.
#define COUNT_VARARGS_IMPL(...) EXPAND(GET_9TH_ARG(__VA_ARGS__))
#define COUNT_VARARGS(...) EXPAND(COUNT_VARARGS_IMPL(__VA_ARGS__, REVERSE_SEQ_8()))

// Dispatches to the correct CAT_WITH_UNDERSCORE_N macro based on the arg count
#define VA_CONCAT_DISPATCHER(count, ...) EXPAND(CONCAT(CAT_WITH_UNDERSCORE_, count)(__VA_ARGS__))
#define VA_CONCAT(...) VA_CONCAT_DISPATCHER(COUNT_VARARGS(__VA_ARGS__), __VA_ARGS__)

std::map<std::string, std::function<bool()>> testCases;

#define START_TESTS void addTests() {
#define END_TESTS }
#define RUN_ALL_TESTS \
addTests(); \
for (const auto& [testName, testFunc] : testCases) { \
    if (!testFunc()) { \
        std::cout << "Test failed: " << testName << std::endl; \
        return 1; \
    } \
} \
return 0;

template <typename Operation, typename=void>
struct TestCaseBuilder;

template <typename Operation>
struct TestCaseBuilder<Operation, std::enable_if_t<IsUnaryType<Operation>::value && std::is_fundamental_v<typename Operation::InputType> && std::is_fundamental_v<typename Operation::OutputType>, void>> {
    template <size_t N>
    static std::function<bool()> build(const std::string& testName,
                                       const std::array<typename Operation::InputType, N>& inputElems,
                                       const std::array<typename Operation::OutputType, N>& expectedElems) {
        return [testName, inputElems, expectedElems]() {
            std::cout << "Running test for " << testName << std::endl;
            bool result{ true };
            for (size_t i = 0; i < N; ++i) {
                const auto generated = Operation::exec(inputElems[i]);
                const auto resultV = generated == expectedElems[i];
                if (!resultV) {
                    std::cout << "Mismatch at test element index " << i
                        << ": Expected value " << expectedElems[i] << ", got " << generated << std::endl;
                }
                result &= resultV;
            }
            if (result) std::cout << "Success!!" << std::endl;
            return result;
            };
    }
};

template <typename Operation>
struct TestCaseBuilder<Operation, std::enable_if_t<IsUnaryType<Operation>::value && (std::is_aggregate_v<typename Operation::InputType> || std::is_aggregate_v<typename Operation::OutputType>), void>> {
    template <size_t N>
    static std::function<bool()> build(const std::string& testName,
                                       const std::array<typename Operation::InputType, N>& inputElems,
                                       const std::array<typename Operation::OutputType, N>& expectedElems) {
        return [testName, inputElems, expectedElems]() -> bool {
            std::cout << "Running test for " << testName << std::endl;
            bool result{ true };
            for (size_t i = 0; i < N; ++i) {
                const auto generated = Operation::exec(inputElems[i]);
                const auto resultV = generated == expectedElems[i];
                const auto arrayGenerated = toArray(generated);
                const auto arrayResult = toArray(resultV);
                for (size_t j = 0; j < cn<decltype(resultV)>; ++j) {
                    if (!arrayResult.at[j]) {
                            std::cout << "Mismatch at test element index " << i << " for vector index " << j
                            << ": Expected value " << toArray(expectedElems[i]).at[j] << ", got " << arrayGenerated.at[j] << std::endl;
                    }
                    result &= arrayResult[j];
                }
            }
            if (result) std::cout << "Success!!" << std::endl;
            return result;
            };
    }
};

#define ADD_UNARY_TEST(ID, OD, UnaryOperation, ...) \
testCases[STRINGIFY(CONCAT(CONCAT(CONCAT(Test, UnaryOperation), _), VA_CONCAT(__VA_ARGS__)))] = \
TestCaseBuilder<UnaryOperation<__VA_ARGS__>>::build(std::string(STRINGIFY(CONCAT(CONCAT(CONCAT(Test, UnaryOperation), _), VA_CONCAT(__VA_ARGS__)))), \
    std::array<typename UnaryOperation<__VA_ARGS__>::InputType, COUNT_VARARGS(DEPAREN(ID))>{DEPAREN(ID)}, \
    std::array<typename UnaryOperation<__VA_ARGS__>::OutputType, COUNT_VARARGS(DEPAREN(OD))>{DEPAREN(OD)});

START_TESTS
ADD_UNARY_TEST((0,200, 100), (0, 200, 1000), SaturateCast, uint, uint)
ADD_UNARY_TEST((minValue<uint1>),(minValue<uint1>),SaturateCast, uint1, uint1)
END_TESTS


// You can add more tests for other type combinations as needed.
int launch() {
    RUN_ALL_TESTS
};