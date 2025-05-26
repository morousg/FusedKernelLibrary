/* Copyright 2024 Oscar Amoros Huguet

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

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/algorithms/basic_ops/cuda_vector.h>
#include <fused_kernel/fused_kernel.h>

#include <iostream>

using namespace fk;

using PtrToTest = Ptr2D<uchar3>;
constexpr int WIDTH = 64;
constexpr int HEIGHT = 64;

PtrToTest test_return_by_value() {
    return PtrToTest(WIDTH, HEIGHT);
}

const PtrToTest& test_return_by_const_reference(const PtrToTest& somePtr) {
    return somePtr;
}

PtrToTest& test_return_by_reference(PtrToTest& somePtr) {
    return somePtr;
}

void test_upload(const cudaStream_t& stream) {
    // Device pointers
    Ptr1D<uchar3> test1D(1333);
    Ptr2D<uchar3> test2D(1333, 444);
    Ptr3D<uchar3> test3D(1333, 444, 22);
    Tensor<uchar3> testTensor(1333, 444, 22);

    // Host Pinned Pointers
    Ptr1D<uchar3> test1D_h(1333, 0, MemType::HostPinned);
    Ptr2D<uchar3> test2D_h(1333, 444, 0, MemType::HostPinned);
    Ptr3D<uchar3> test3D_h(1333, 444, 22, 1, 0, MemType::HostPinned);
    Tensor<uchar3> testTensor_h(1333, 444, 22, 1, MemType::HostPinned);

    // Must work
    test1D_h.upload(test1D, stream);
    test2D_h.upload(test2D, stream);
    test3D_h.upload(test3D, stream);
    testTensor_h.upload(testTensor, stream);

    // Must not work
    try {
        test1D.upload(test1D_h);
    } catch (const std::exception& e) {
        std::cout << "Expected exception: " << e.what() << std::endl;
    }

    // Compile time error
    // test2D_h.upload(test3D);
}

void test_download(const cudaStream_t& stream) {
    // Device pointers
    Ptr1D<uchar3> test1D(1333);
    Ptr2D<uchar3> test2D(1333, 444);
    Ptr3D<uchar3> test3D(1333, 444, 22);
    Tensor<uchar3> testTensor(1333, 444, 22);

    // Host Pinned Pointers
    Ptr1D<uchar3> test1D_h(1333, 0, MemType::HostPinned);
    Ptr2D<uchar3> test2D_h(1333, 444, 0, MemType::HostPinned);
    Ptr3D<uchar3> test3D_h(1333, 444, 22, 1, 0, MemType::HostPinned);
    Tensor<uchar3> testTensor_h(1333, 444, 22, 1, MemType::HostPinned);

    // Must work
    test1D.download(test1D_h, stream);
    test2D.download(test2D_h, stream);
    test3D.download(test3D_h, stream);
    testTensor.download(testTensor_h, stream);

    // Must not work
    try {
        test1D_h.download(test1D);
    } catch (const std::exception& e) {
        std::cout << "Expected exception: " << e.what() << std::endl;
    }
}

int launch() {

    cudaStream_t stream;
    gpuErrchk(cudaStreamCreate(&stream));

    PtrToTest test0(WIDTH, HEIGHT, 0, MemType::HostPinned);
    setTo(make_<uchar3>(1, 2, 3), test0);
    bool h_correct{ true };
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            const bool3 boolVect = *PtrAccessor<_2D>::cr_point(Point(x, y), test0.ptr()) == make_<uchar3>(1, 2, 3);
            h_correct &= VectorAnd<bool3>::exec(boolVect);
        }
    }

    PtrToTest test1(WIDTH, HEIGHT);

    auto test2 = PtrToTest(WIDTH, HEIGHT);

    PtrToTest test3;
    test3 = PtrToTest(WIDTH, HEIGHT);

    auto test4 = test_return_by_value();
    PtrToTest somePtr(WIDTH, HEIGHT);
    const PtrToTest& test5 = test_return_by_const_reference(somePtr);
    PtrToTest& test6 = test_return_by_reference(somePtr);

    bool result = test1.getRefCount() == 1;
    result &= test2.getRefCount() == 1;
    result &= test3.getRefCount() == 1;
    result &= test4.getRefCount() == 1;
    result &= test5.getRefCount() == 1;
    result &= test6.getRefCount() == 1;

    PtrToTest test7(WIDTH, HEIGHT);
    PtrToTest h_test7(WIDTH, HEIGHT, 0, MemType::HostPinned);
    setTo(make_<uchar3>(3,6,10), test7, stream);
    gpuErrchk(cudaMemcpy2DAsync(h_test7.ptr().data, h_test7.ptr().dims.pitch,
                                test7.ptr().data, test7.ptr().dims.pitch,
                                WIDTH * sizeof(uchar3), HEIGHT, cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));

    bool h_correct2{ true };
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            const bool3 boolVect = *PtrAccessor<_2D>::cr_point(Point(x, y), h_test7.ptr()) == make_<uchar3>(3, 6, 10);
            h_correct2 &= VectorAnd<bool3>::exec(boolVect);
        }
    }

    test_upload(stream);
    test_download(stream);

    return result && h_correct && h_correct2 ? 0 : -1;
}
