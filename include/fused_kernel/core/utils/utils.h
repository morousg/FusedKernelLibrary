/* Copyright 2023-2024 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_UTILS
#define FK_UTILS

#include <string>
#include <stdexcept>
#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#include <cuda.h>
#include <cuda_runtime.h>

#define FK_DEVICE_FUSE static constexpr __device__ __forceinline__
#define FK_DEVICE_CNST constexpr __device__ __forceinline__
#define FK_HOST_DEVICE_FUSE FK_DEVICE_FUSE __host__
#define FK_HOST_DEVICE_CNST FK_DEVICE_CNST __host__
#define FK_HOST_FUSE static constexpr __forceinline__ __host__
#define FK_HOST_CNST constexpr __forceinline__ __host__
#define FK_HOST_STATIC static __forceinline__ __host__
#define FK_HOST_DEVICE_STATIC static __forceinline__ __host__ __device__
#define FK_RESTRICT __restrict__
#else
#define FK_DEVICE_FUSE static constexpr inline
#define FK_DEVICE_CNST constexpr inline
#define FK_HOST_DEVICE_FUSE FK_DEVICE_FUSE
#define FK_HOST_DEVICE_CNST FK_DEVICE_CNST
#define FK_HOST_FUSE static constexpr inline
#define FK_HOST_CNST constexpr inline
#define FK_HOST_STATIC static inline
#define FK_HOST_DEVICE_STATIC static inline
#ifdef _MSC_VER
#define FK_RESTRICT __restrict
#else
#define FK_RESTRICT __restrict__
#endif
#endif

#ifdef CUDART_VERSION
#define CUDART_MAJOR_VERSION CUDART_VERSION/1000
#else
#define CUDART_MAJOR_VERSION 0 // We are not compiling with nvcc
#endif

#define FK_STATIC_STRUCT(struct_name) \
    public: /* Ensure deletions are in a public section (conventional) */ \
        struct_name() = delete; \
        struct_name(const struct_name&) = delete; \
        struct_name& operator=(const struct_name&) = delete; \
        struct_name(struct_name&&) = delete; \
        struct_name& operator=(struct_name&&) = delete;

#define FK_STATIC_STRUCT_CHILD(struct_name, struct_alias) \
    public: /* Ensure deletions are in a public section (conventional) */ \
        struct_name() = delete; \
        struct_name(const struct_alias&) = delete; \
        struct_name& operator=(const struct_alias&) = delete; \
        struct_name(struct_alias&&) = delete; \
        struct_name& operator=(struct_alias&&) = delete;

using uchar = unsigned char;
using schar = signed char;
using uint = unsigned int;
using longlong = long long;
using ulonglong = unsigned long long;

using ushort = unsigned short;
using ulong = unsigned long;

#if defined(__NVCC__) || defined(__HIP__)
namespace fk {
    inline void gpuAssert(cudaError_t code,
        const char *file,
        int line,
        bool abort = true) {
        if (code != cudaSuccess) {
            std::string message = "GPUassert: ";
            message.append(cudaGetErrorString(code));
            message.append(" File: ");
            message.append(file);
            message.append(" Line:");
            message.append(std::to_string(line).c_str());
            message.append("/n");
            if (abort) throw std::runtime_error(message.c_str());
        }
    }
} // namespace fk

#define gpuErrchk(ans) { fk::gpuAssert((ans), __FILE__, __LINE__, true); }
#endif
// Null type, used for Operation required aliases that can not still be known, because they are deduced
// from a backwards operation that is till not defined.
struct NullType {};

#endif
