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

// In the future we will change all .h files to .h files, since we are going to be multi platform
#ifndef FK_EXECUTORS_CUH
#define FK_EXECUTORS_CUH

#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/execution_model/data_parallel_patterns.h>
#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/set.h>

namespace fk {

    enum class DPPType {
        Transform,
        Reduce,
        MMAD,
        Convolution3D,
        None
    };

    struct CtxDim3 {
        uint x;
        uint y;
        uint z;
        constexpr CtxDim3(const dim3& dims) : x(dims.x), y(dims.y), z(dims.z) {}
        constexpr CtxDim3() : x(1), y(1), z(1) {}
        constexpr CtxDim3(const uint& x) : x(x), y(1), z(1) {}
        constexpr CtxDim3(const uint& x, const uint& y) : x(x), y(y), z(1) {}
        constexpr CtxDim3(const uint& x, const uint& y, const uint& z) : x(x), y(y), z(z) {}
    };

    template <enum ParArch PA, enum DPPType DPP, bool THREAD_COARSENING = false>
    struct Executor {
        static_assert(PA == ParArch::GPU_NVIDIA, "Only CUDA is supported for now");
        static_assert(DPP == DPPType::Transform, "Only Transform is supported for now");
    };

    struct ComputeBestSolutionBase {
        static constexpr ParArch PA = ParArch::GPU_NVIDIA;
        FK_HOST_FUSE uint computeDiscardedThreads(const uint width, const uint height, const uint blockDimx, const uint blockDimy) {
            const uint modX = width % blockDimx;
            const uint modY = height % blockDimy;
            const uint th_disabled_in_X = modX == 0 ? 0 : blockDimx - modX;
            const uint th_disabled_in_Y = modY == 0 ? 0 : blockDimy - modY;
            return (th_disabled_in_X * (modY == 0 ? height : (height + blockDimy)) + th_disabled_in_Y * width);
        }
    };

    template <uint bxS_t, uint byS_t>
    struct computeBestSolution {};

    template <uint bxS_t>
    struct computeBestSolution<bxS_t, 0> final : public ComputeBestSolutionBase {
        FK_HOST_FUSE void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint(&blockDimX)[4], const uint(&blockDimY)[2][4]) {
            const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[bxS_t], blockDimY[0][bxS_t]);
            if (minDiscardedThreads > currentDiscardedThreads) {
                minDiscardedThreads = currentDiscardedThreads;
                bxS = bxS_t;
                byS = 0;
                if (minDiscardedThreads == 0) return;
            }
            computeBestSolution<bxS_t, 1>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);
        }
    };

    template <uint bxS_t>
    struct computeBestSolution<bxS_t, 1> final : public ComputeBestSolutionBase{
        FK_HOST_FUSE void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint(&blockDimX)[4], const uint(&blockDimY)[2][4]) {
            const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[bxS_t], blockDimY[1][bxS_t]);
            if (minDiscardedThreads > currentDiscardedThreads) {
                minDiscardedThreads = currentDiscardedThreads;
                bxS = bxS_t;
                byS = 1;
                if constexpr (bxS_t == 3) return;
                if (minDiscardedThreads == 0) return;
            }
            computeBestSolution<bxS_t + 1, 0>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);
        }
    };

    template <>
    struct computeBestSolution<3, 1> final : public ComputeBestSolutionBase {
        FK_HOST_FUSE void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint(&blockDimX)[4], const uint(&blockDimY)[2][4]) {
            const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[3], blockDimY[1][3]);
            if (minDiscardedThreads > currentDiscardedThreads) {
                minDiscardedThreads = currentDiscardedThreads;
                bxS = 3;
                byS = 1;
            }
        }
    };

    template <bool THREAD_COARSENING>
    struct Executor<ParArch::GPU_NVIDIA, DPPType::Transform, THREAD_COARSENING> {
    private:
        FK_HOST_FUSE CtxDim3 getDefaultBlockSize(const uint& width, const uint& height) {
            constexpr uint blockDimX[4] = { 32, 64, 128, 256 };  // Possible block sizes in the x axis
            constexpr uint blockDimY[2][4] = { { 8,  4,   2,   1},
                                              { 6,  3,   3,   2} };  // Possible block sizes in the y axis according to blockDim.x

            uint minDiscardedThreads = UINT_MAX;
            uint bxS = 0; // from 0 to 3
            uint byS = 0; // from 0 to 1

            computeBestSolution<0, 0>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);

            return CtxDim3(blockDimX[bxS], blockDimY[byS][bxS]);
        }

        template <typename... IOps>
        FK_HOST_FUSE void executeOperations_helper(const cudaStream_t& stream, const IOps&... iOps) {
            constexpr bool THREAD_FUSION = THREAD_COARSENING;
            constexpr ParArch PA = ParArch::GPU_NVIDIA;
            const auto tDetails = TransformDPP<PA, void>::build_details<THREAD_FUSION>(iOps...);
            if constexpr (decltype(tDetails)::TFI::ENABLED) {
                const ActiveThreads activeThreads = tDetails.activeThreads;

                const CtxDim3 ctx_block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

                const dim3 block{ ctx_block.x, ctx_block.y, 1 };
                const dim3 grid{ static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                                 static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))),
                                 activeThreads.z };
                if (!tDetails.threadDivisible) {
                    launchTransformDPP_Kernel<PA,false> << <grid, block, 0, stream >> > (tDetails, iOps...);
                    gpuErrchk(cudaGetLastError());
                } else {
                    launchTransformDPP_Kernel<PA, true> << <grid, block, 0, stream >> > (tDetails, iOps...);
                    gpuErrchk(cudaGetLastError());
                }
            } else {
                const auto readOp = get<0>(iOps...);

                const ActiveThreads activeThreads = readOp.getActiveThreads();

                const CtxDim3 ctx_block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

                const dim3 block{ ctx_block.x, ctx_block.y, 1 };
                const dim3 grid{ static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                                 static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))),
                                 activeThreads.z };
                launchTransformDPP_Kernel<PA, true><<<grid, block, 0, stream>>>(tDetails, iOps...);
                gpuErrchk(cudaGetLastError());
            }
        }
    public:
        template <typename... IOps>
        FK_HOST_FUSE void executeOperations(const cudaStream_t& stream, const IOps&... iOps) {
            executeOperations_helper(stream, iOps...);
        }

        template <typename I, typename... IOps>
        FK_HOST_FUSE void executeOperations(const Ptr2D<I>& input, const cudaStream_t& stream,
                                            const IOps&... iOps) {
            executeOperations_helper(stream, PerThreadRead<_2D, I>::build({ input }), iOps...);
        }

        template <typename I, typename O, typename... IOps>
        FK_HOST_FUSE void executeOperations(const Ptr2D<I>& input, const Ptr2D<O>& output,
                                            const cudaStream_t& stream, const IOps&... iOps) {
            executeOperations_helper(stream,
                PerThreadRead<_2D, I>::build({ input }), iOps..., PerThreadWrite<_2D, O>::build({ output }));
        }

        template <typename I, size_t BATCH, typename... IOps>
        FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, BATCH>& input, const int& activeBatch, const I& defaultValue,
                                            const cudaStream_t& stream, const IOps&... iOps) {
            const auto batchReadIOp = PerThreadRead<_2D, I>::build(activeBatch, defaultValue, input);
            executeOperations_helper(stream, batchReadIOp, iOps...);
        }

        template <typename I, size_t BATCH, typename... IOps>
        FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, BATCH>& input,
                                            const cudaStream_t& stream, const IOps&... iOps) {
            const auto batchReadIOp = PerThreadRead<_2D, I>::build(input);
            executeOperations_helper(stream, batchReadIOp, iOps...);
        }

        template <typename I, typename O, size_t Batch, typename... IOps>
        FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch, const I& defaultValue,
                                            const Tensor<O>& output, const cudaStream_t& stream, const IOps&... iOps) {
            const auto batchReadIOp = PerThreadRead<_2D, I>::build(activeBatch, defaultValue, input);
            const auto writeOp = PerThreadWrite<_3D, O>::build(output);
            executeOperations_helper(stream, batchReadIOp, iOps..., writeOp);
        }

        template <typename I, typename O, size_t Batch, typename... IOps>
        FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const Tensor<O>& output,
                                            const cudaStream_t& stream, const IOps&... iOps) {
            const auto batchReadIOp = PerThreadRead<_2D, I>::build(input);
            const auto writeOp = PerThreadWrite<_3D, O>::build(output);
            executeOperations_helper(stream, batchReadIOp, iOps..., writeOp);
        }
    };

    template <ND D, typename T>
    inline constexpr void setTo(const T& value, Ptr<D, T>& outputPtr, const cudaStream_t& stream = 0) {
        RawPtr<D, T> output = outputPtr.ptr();
        if (outputPtr.getMemType() == MemType::Device) {
            if constexpr (D == _1D) {
                const ActiveThreads activeThreads(output.dims.width);
                Executor<ParArch::GPU_NVIDIA, DPPType::Transform>::executeOperations(stream, ReadSet<T>::build(value, activeThreads), PerThreadWrite<D, T>::build({ output }));
            } else if constexpr (D == _2D) {
                const ActiveThreads activeThreads(output.dims.width, output.dims.height);
                Executor<ParArch::GPU_NVIDIA, DPPType::Transform>::executeOperations(stream, ReadSet<T>::build(value, activeThreads), PerThreadWrite<D, T>::build({ output }));
            } else if constexpr (D == _3D) {
                const ActiveThreads activeThreads(output.dims.width, output.dims.height, output.dims.planes);
                Executor<ParArch::GPU_NVIDIA, DPPType::Transform>::executeOperations(stream, ReadSet<T>::build(value, activeThreads), PerThreadWrite<D, T>::build({ output }));
            }
        } else {
            for (int i = 0; i < (int)outputPtr.getNumElements(); i++) {
                output.data[i] = value;
            }
        }
    }
} // namespace fk

#endif // FK_EXECUTORS_CUH