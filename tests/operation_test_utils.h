/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Albert Andaluz
   Copyright 2025 Grup Mediapro S.L.U

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_OPERATION_TEST_UTILS_H
#define FK_OPERATION_TEST_UTILS_H

#include <map>
#include <functional>
#include <iostream>

#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/core/execution_model/executors.h>

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

#define START_ADDING_TESTS void addTests() {
#define STOP_ADDING_TESTS }
#define RUN_ALL_TESTS \
addTests(); \
bool correct{true}; \
for (const auto& [testName, testFunc] : testCases) { \
    if (!testFunc()) { \
        correct = false; \
    } \
} \
return correct ? 0 : -1;

template <typename Operation, typename = void>
struct TestCaseBuilder;

namespace test_case_builder::detail {
    template <typename Operation, size_t N>
    fk::Ptr1D<typename Operation::OutputType>
    launchUnary(const std::string& testName,
                const std::array<typename Operation::InputType, N>& inputElems) {
        fk::Stream stream;
        using I = typename Operation::InputType;
        using O = typename Operation::OutputType;
        fk::Ptr1D<I> inputPtr(N);
        fk::Ptr1D<O> outputPtr(N);
        for (size_t i = 0; i < N; ++i) {
            inputPtr.at(fk::Point(i)) = inputElems[i];
        }
        std::cout << "Running test for " << "\033[1;33m" <<testName << "\033[1;33m" << ": ";
        inputPtr.upload(stream);
        fk::Executor<fk::TransformDPP<>>::executeOperations(inputPtr, outputPtr, stream,
                                                            Operation::build());
        outputPtr.download(stream);
        stream.sync();
        return outputPtr;
    }
}

template <typename Operation>
struct TestCaseBuilder<Operation, std::enable_if_t<fk::IsUnaryType<Operation>::value &&
                                    std::is_fundamental_v<typename Operation::InputType> &&
                                    std::is_fundamental_v<typename Operation::OutputType>, void>> {
    template <size_t N>
    static std::function<bool()> build(const std::string& testName,
        const std::array<typename Operation::InputType, N>& inputElems,
        const std::array<typename Operation::OutputType, N>& expectedElems) {
        return [testName, inputElems, expectedElems]() {
            const auto outputPtr = test_case_builder::detail::launchUnary<Operation>(testName, inputElems);
            bool result{ true };
            for (size_t i = 0; i < N; ++i) {
                const auto generated = outputPtr.at(fk::Point(i));
                const auto resultV = generated == expectedElems[i];
                if (!resultV) {
                    std::cout << "\033[32m" << "FAIL!!" << "\033[0m" std::endl;
                    std::cout << std::endl<< "\033[31m Mismatch at test element index " << i << ": Expected value "
                              << expectedElems[i] << ", got " << generated << "\033[0m"<< std::endl;
                }
                result &= resultV;
            }
            if (result)
                std::cout << "\033[32m" << "Success!!" << "\033[0m" std::endl;
            return result;
            };
    }
};

template <typename Operation>
struct TestCaseBuilder<Operation, std::enable_if_t<fk::IsUnaryType<Operation>::value &&
                                    (std::is_aggregate_v<typename Operation::InputType> ||
                                     std::is_aggregate_v<typename Operation::OutputType>), void>> {
    template <size_t N>
    static std::function<bool()> build(const std::string& testName,
        const std::array<typename Operation::InputType, N>& inputElems,
        const std::array<typename Operation::OutputType, N>& expectedElems) {
        return [testName, inputElems, expectedElems]() -> bool {
            const auto outputPtr = test_case_builder::detail::launchUnary<Operation>(testName, inputElems);
            bool result{ true };
            for (size_t i = 0; i < N; ++i) {
                const auto generated = outputPtr.at(fk::Point(i));
                const auto resultV = generated == expectedElems[i];
                const auto arrayGenerated = fk::toArray(generated);
                const auto arrayResult = fk::toArray(resultV);
                for (size_t j = 0; j < fk::cn<decltype(resultV)>; ++j) {
                    if (!arrayResult.at[j]) {
                        std::cout << "\033[31m" << "FAIL!!" << "\033[31m" <<std::endl;
                        std::cout <<"\033[31m"<< "Mismatch at test element index " << i << " for vector index " << j
                            << ": Expected value " << fk::toArray(expectedElems[i]).at[j] << ", got " << arrayGenerated.at[j] << "\033[0m"<<std::endl;
                    }
                    result &= arrayResult[j];
                }
            }
            if (result) 
                  std::cout << "\033[32m" << "Success!!" << "\033[0m" <<std::endl;
            return result;
            };
    }
};

#define ADD_UNARY_TEST(ID, OD, UnaryOperation, ...) \
testCases[STRINGIFY(CONCAT(CONCAT(CONCAT(Test, UnaryOperation), _), VA_CONCAT(__VA_ARGS__)))] = \
TestCaseBuilder<UnaryOperation<__VA_ARGS__>>::build(std::string(STRINGIFY(CONCAT(CONCAT(CONCAT(Test, UnaryOperation), _), VA_CONCAT(__VA_ARGS__)))), \
    std::array<typename UnaryOperation<__VA_ARGS__>::InputType, COUNT_VARARGS(DEPAREN(ID))>{DEPAREN(ID)}, \
    std::array<typename UnaryOperation<__VA_ARGS__>::OutputType, COUNT_VARARGS(DEPAREN(OD))>{DEPAREN(OD)});

#endif
