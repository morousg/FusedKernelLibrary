/* Copyright 2023-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <tests/main.h>

#include <iostream>

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/core/utils/template_operations.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/algorithms/basic_ops/cuda_vector.h>

template <typename T>
bool testPtr_2D() {
    constexpr size_t width = 1920;
    constexpr size_t height = 1080;
    constexpr size_t width_crop = 300;
    constexpr size_t height_crop = 200;

    fk::Point startPoint = {100, 200};

    fk::Stream stream;

    fk::Ptr2D<T> input(width, height);
    fk::setTo(fk::make_set<T>(2), input, stream);
    fk::Ptr2D<T> cropedInput = input.crop(startPoint, fk::PtrDims<fk::_2D>(width_crop, height_crop));
    fk::Ptr2D<T> output(width_crop, height_crop);
    fk::Ptr2D<T> outputBig(width, height);

    fk::Read<fk::PerThreadRead<fk::_2D, T>> readCrop{{cropedInput}};
    fk::Read<fk::PerThreadRead<fk::_2D, T>> readFull{{input}};

    fk::WriteInstantiableOperation<fk::PerThreadWrite<fk::_2D, T>> opFinal_2D = { {output} };
    fk::WriteInstantiableOperation<fk::PerThreadWrite<fk::_2D, T>> opFinal_2DBig = { {outputBig} };

    for (int i=0; i<100; i++) {
        fk::executeOperations(stream, readCrop, opFinal_2D);
        fk::executeOperations(stream, readFull, opFinal_2DBig);
    }

    stream.sync();

    for (int y = 0; y < output.dims().height; ++y) {
        for (int x = 0; x < output.dims().width; ++x) {
            const fk::VectorType_t<bool, fk::cn<T>> result = output.at({ x, y }) != fk::make_set<T>(2);
            if (fk::vecAnd(result)) {
                std::cout << "Error in output at (" << x << ", " << y << "): " << output.at({ x, y }) << std::endl;
                return false;
            }
        }
    }

    // TODO: use some values and check results correctness

    return true;
}

int launch() {
    bool test2Dpassed = true;

    test2Dpassed &= testPtr_2D<uchar>();
    test2Dpassed &= testPtr_2D<uchar3>();
    test2Dpassed &= testPtr_2D<float>();
    test2Dpassed &= testPtr_2D<float3>();

    fk::Stream stream;

    fk::Ptr2D<uchar> input(64,64);
    fk::Ptr2D<uint> output(64,64);

    fk::Read<fk::PerThreadRead<fk::_2D, uchar>> read{ {input} };
    fk::Unary<fk::SaturateCast<uchar, uint>> cast = {};
    fk::Write<fk::PerThreadWrite<fk::_2D, uint>> write { {output} };

    auto fusedDF = fk::fuse(read, cast, fk::Binary<fk::Mul<uint>>{4});
    static_assert(std::is_same_v<decltype(fusedDF.params.instance.params), fk::RawPtr<fk::_2D, uchar>>, "Unexpected type for params");
    //fusedDF.params.next.instance.params; // Should not compile
    static_assert(std::is_same_v<decltype(fusedDF.params.next.next.instance.params), uint>, "Unexpected type for params");

    fk::executeOperations(stream, fusedDF, write);
    stream.sync();

    fk::OperationTuple<fk::PerThreadRead<fk::_2D, uchar>, fk::SaturateCast<uchar, uint>, fk::PerThreadWrite<fk::_2D, uint>> myTup{};

    fk::get<2>(myTup);
    constexpr bool test1 = std::is_same_v<fk::get_type_t<0, decltype(myTup)>, fk::PerThreadRead<fk::_2D, uchar>>;
    constexpr bool test2 = std::is_same_v<fk::get_type_t<1, decltype(myTup)>, fk::SaturateCast<uchar, uint>>;
    constexpr bool test3 = std::is_same_v<fk::get_type_t<2, decltype(myTup)>, fk::PerThreadWrite<fk::_2D, uint>>;

    if (test2Dpassed && fk::and_v<test1, test2, test3>) {
        std::cout << "cuda_transform executed!!" << std::endl;
        return 0;
    } else {
        std::cout << "cuda_transform failed!!" << std::endl;
        return -1;
    }
}