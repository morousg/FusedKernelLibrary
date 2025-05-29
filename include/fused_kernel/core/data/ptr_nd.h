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

#ifndef FK_PTR_ND_H
#define FK_PTR_ND_H

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/data/size.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/core/data/nd.h>

namespace fk {
	enum MemType { Device, Host, HostPinned, DeviceAndPinned };
#if defined(__NVCC__) || defined(__HIP__)
    constexpr MemType defaultMemType = DeviceAndPinned;
#else
    constexpr MemType defaultMemType = Host;
#endif

    template <enum ND D>
    struct PtrDims;

    template <>
    struct PtrDims<_1D> {
        uint width{0};
        uint pitch{0};

        FK_HOST_DEVICE_CNST
            PtrDims<_1D>() {}
        FK_HOST_DEVICE_CNST
            PtrDims<_1D>(uint width_, uint pitch_ = 0) : width(width_), pitch(pitch_) {}
    };

    template <>
    struct PtrDims<_2D> {
        uint width{ 0 };
        uint height{ 0 };
        uint pitch{ 0 };

        FK_HOST_DEVICE_CNST PtrDims<_2D>() {}
        FK_HOST_DEVICE_CNST PtrDims<_2D>(uint width_, uint height_, uint pitch_ = 0) :
            width(width_), height(height_), pitch(pitch_) {}
    };

    template <>
    struct PtrDims<_3D> {
        // Image batch shape
        // R,G,B
        // R,G,B
        // R,G,B

        // Width and Height of one individual image
        uint width{0};
        uint height{0};
        // Number of images
        uint planes{0};
        // Number of color channels
        uint color_planes{0};

        // Pitch for each image
        uint pitch{0};

        // Pitch to jump one plane
        uint plane_pitch{0};

        FK_HOST_DEVICE_CNST PtrDims<_3D>() {}
        FK_HOST_DEVICE_CNST
            PtrDims<_3D>(uint width_, uint height_, uint planes_, uint color_planes_ = 1, uint pitch_ = 0) :
            width(width_), height(height_), planes(planes_), color_planes(color_planes_),
            pitch(pitch_), plane_pitch(pitch_ * height_) {}
    };

    template <>
    struct PtrDims<T3D> {
        // Image batch shape
        // R,R,R
        // G,G,G
        // B,B,B

        // Width and Height of one individual image
        uint width{ 0 };
        uint height{ 0 };
        // Number of images
        uint planes{ 0 };
        // Number of color channels
        uint color_planes{ 0 };

        // Pitch for each image
        uint pitch{ 0 };

        // Pitch to jump one plane
        uint plane_pitch{ 0 };

        // Pitch to jump to the next plane of the same image
        uint color_planes_pitch{ 0 };

        FK_HOST_DEVICE_CNST PtrDims<T3D>() {}
        FK_HOST_DEVICE_CNST
            PtrDims<T3D>(uint width_, uint height_, uint planes_, uint color_planes_ = 1) :
            width(width_), height(height_), planes(planes_), color_planes(color_planes_),
            pitch(0), plane_pitch(0), color_planes_pitch(0) {}
    };

    template <int W>
    struct StaticPtrDims1D {
        static constexpr uint width{ W };
    };

    template <int W, int H>
    struct StaticPtrDims2D {
        static constexpr uint width{ W };
        static constexpr uint height{ H };
    };

    template<int W, int H, int P>
    struct StaticPtrDims3D {
        static constexpr uint width{ W };
        static constexpr uint height{ H };
        static constexpr uint planes{ P };
    };

    template <enum ND D, typename T>
    struct RawPtr;

    template <typename T>
    struct RawPtr<_1D, T> {
        T* data{nullptr};
        PtrDims<_1D> dims;
        using type = T;
        enum { ND = _1D };
    };

    template <typename T>
    struct RawPtr<_2D, T> {
        T* data{nullptr};
        PtrDims<_2D> dims;
        using type = T;
        enum { ND = _2D };
    };

    template <typename T>
    struct RawPtr<_3D, T> {
        T* data{nullptr};
        PtrDims<_3D> dims;
        using type = T;
        enum { ND = _3D };
    };

    template <typename T>
    struct RawPtr<T3D, T> {
        T* data{nullptr};
        PtrDims<T3D> dims;
        using type = T;
        enum { ND = T3D };
    };

    template<typename Dims, typename T>
    struct StaticRawPtr;

    template<typename T, int W>
    struct StaticRawPtr<StaticPtrDims1D<W>, T> {
        using type = T;
        T data[W];
        static constexpr StaticPtrDims1D<W> dims{};
        static constexpr ND nd{ _1D };
    };

    template<typename T, int W, int H>
    struct StaticRawPtr<StaticPtrDims2D<W, H>, T> {
        using type = T;
        T data[H][W];
        static constexpr StaticPtrDims2D<W, H> dims{};
        static constexpr ND nd{ _2D };
    };

    template<typename T, int W, int H, int P>
    struct StaticRawPtr<StaticPtrDims3D<W, H, P>, T> {
        using type = T;
        T data[P][H][W];
        static constexpr StaticPtrDims3D<W, H, P> dims{};
        static constexpr ND nd{ _3D };
    };

    template <enum ND D>
    struct PtrAccessor;

    template <>
    struct PtrAccessor<_1D> {
        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_FUSE const BiggerType* cr_point(const Point& p, const RawPtr<_1D, T>& ptr) {
            return ((const BiggerType*)ptr.data) + p.x;
        }

        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_STATIC BiggerType* point(const Point& p, const RawPtr<_1D, T>& ptr) {
            return (BiggerType*)ptr.data + p.x;
        }
    };

    template <>
    struct PtrAccessor<_2D> {
        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_FUSE const BiggerType* cr_point(const Point& p, const RawPtr<_2D, T>& ptr) {
            return (const BiggerType*)((const char*)ptr.data + (p.y * ptr.dims.pitch)) + p.x;
        }

        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_STATIC BiggerType* point(const Point& p, const RawPtr<_2D, T>& ptr) {
            return (BiggerType*)((char*)ptr.data + (p.y * ptr.dims.pitch)) + p.x;
        }
    };

    template <>
    struct PtrAccessor<_3D> {
        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_FUSE const BiggerType* cr_point(const Point& p, const RawPtr<_3D, T>& ptr) {
            return (const BiggerType*)((const char*)ptr.data + (ptr.dims.plane_pitch * ptr.dims.color_planes * p.z) + (p.y * ptr.dims.pitch)) + p.x;
        }

        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_STATIC BiggerType* point(const Point& p, const RawPtr<_3D, T>& ptr) {
            return (BiggerType*)((char*)ptr.data + (ptr.dims.plane_pitch * ptr.dims.color_planes * p.z) + (p.y * ptr.dims.pitch)) + p.x;
        }
    };

    template <>
    struct PtrAccessor<T3D> {
        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_FUSE const BiggerType* cr_point(const Point& p, const RawPtr<T3D, T>& ptr, const uint& color_plane = 0) {
            return (const BiggerType*)((const char*)ptr.data + (color_plane * ptr.dims.color_planes_pitch) + (ptr.dims.plane_pitch * p.z) + (ptr.dims.pitch * p.y)) + p.x;
        }

        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_STATIC BiggerType* point(const Point& p, const RawPtr<T3D, T>& ptr, const uint& color_plane = 0) {
            return (BiggerType*)((char*)ptr.data + (color_plane * ptr.dims.color_planes_pitch) + (ptr.dims.plane_pitch * p.z) + (ptr.dims.pitch * p.y)) + p.x;
        }
    };

    template<enum ND D>
    struct StaticPtrAccessor;

    template<>
    struct StaticPtrAccessor<_1D> {
        template <int W, typename T>
        FK_HOST_DEVICE_FUSE T read(const Point& p, const StaticRawPtr<StaticPtrDims1D<W>, T>& ptr) {
            return ptr.data[p.x];
        }

        template <int W, typename T>
        FK_HOST_DEVICE_FUSE void write(const Point& p, StaticRawPtr<StaticPtrDims1D<W>, T>& ptr, const T& value) {
            ptr.data[p.x] = value;
        }
    };

    template<>
    struct StaticPtrAccessor<_2D> {
        template <int W, int H, typename T>
        FK_HOST_DEVICE_FUSE T read(const Point& p, const StaticRawPtr<StaticPtrDims2D<W, H>, T>& ptr) {
            return ptr.data[p.y][p.x];
        }

        template <int W, int H, typename T>
        FK_HOST_DEVICE_FUSE void write(const Point& p, StaticRawPtr<StaticPtrDims2D<W, H>, T>& ptr, const T& value) {
            ptr.data[p.y][p.x] = value;
        }
    };

    template<>
    struct StaticPtrAccessor<_3D> {
        template <int W, int H, int P, typename T>
        FK_HOST_DEVICE_FUSE T read(const Point& p, const StaticRawPtr<StaticPtrDims3D<W, H, P>, T>& ptr) {
            return ptr.data[p.z][p.y][p.x];
        }

        template <int W, int H, int P, typename T>
        FK_HOST_DEVICE_FUSE void write(const Point& p, StaticRawPtr<StaticPtrDims3D<W, H, P>, T>& ptr, const T& value) {
            ptr.data[p.z][p.y][p.x] = value;
        }
    };

    template <typename StaticRawPtr>
    struct StaticPtr {
        using Type = typename StaticRawPtr::type;
        using At = StaticPtrAccessor<StaticRawPtr::ND>;
        StaticRawPtr ptr_a;
        inline constexpr StaticRawPtr ptr() const {
            return ptr_a;
        }
        inline constexpr auto dims() const {
            return ptr_a.dims;
        }
    };

    template <enum ND D, typename T>
    struct PtrImpl;

    template <typename T>
    struct PtrImpl<_1D, T> {
        FK_HOST_FUSE size_t sizeInBytes(const PtrDims<_1D>& dims) {
            return dims.pitch;
        }
        FK_HOST_FUSE uint getNumElements(const PtrDims<_1D>& dims) {
            return dims.width;
        }
        FK_HOST_FUSE void d_malloc(RawPtr<_1D, T>& ptr_a) {
            gpuErrchk(cudaMalloc(&ptr_a.data, sizeof(T) * ptr_a.dims.width));
            ptr_a.dims.pitch = sizeof(T) * ptr_a.dims.width;
        }
        FK_HOST_FUSE void h_malloc_init(PtrDims<_1D>& dims) {
            dims.pitch = sizeof(T) * dims.width;
        }
    };

    template <typename T>
    struct PtrImpl<_2D, T> {
        FK_HOST_FUSE size_t sizeInBytes(const PtrDims<_2D>& dims) {
            return dims.pitch * dims.height;
        }
        FK_HOST_FUSE uint getNumElements(const PtrDims<_2D>& dims) {
            return dims.width * dims.height;
        }
        FK_HOST_FUSE void d_malloc(RawPtr<_2D, T>& ptr_a) {
            if (ptr_a.dims.pitch == 0) {
                size_t pitch;
                gpuErrchk(cudaMallocPitch(&ptr_a.data, &pitch, sizeof(T) * ptr_a.dims.width, ptr_a.dims.height));
                ptr_a.dims.pitch = static_cast<int>(pitch);
            } else {
                gpuErrchk(cudaMalloc(&ptr_a.data, PtrImpl<_2D, T>::sizeInBytes(ptr_a.dims)));
            }
        }
        FK_HOST_FUSE void h_malloc_init(PtrDims<_2D>& dims) {
            dims.pitch = sizeof(T) * dims.width;
        }
    };

    template <typename T>
    struct PtrImpl<_3D, T> {
        FK_HOST_FUSE size_t sizeInBytes(const PtrDims<_3D>& dims) {
            return dims.pitch * dims.height * dims.planes * dims.color_planes;
        }
        FK_HOST_FUSE uint getNumElements(const PtrDims<_3D>& dims) {
            return dims.width * dims.height * dims.planes * dims.color_planes;
        }
        FK_HOST_FUSE void d_malloc(RawPtr<_3D, T>& ptr_a) {
            if (ptr_a.dims.pitch == 0) {
                ptr_a.dims.pitch = sizeof(T) * ptr_a.dims.width;
            }
            gpuErrchk(cudaMalloc(&ptr_a.data, PtrImpl<_3D, T>::sizeInBytes(ptr_a.dims)));
            ptr_a.dims.plane_pitch = ptr_a.dims.pitch * ptr_a.dims.height;
        }
        FK_HOST_FUSE void h_malloc_init(PtrDims<_3D>& dims) {
            dims.pitch = sizeof(T) * dims.width;
            dims.plane_pitch = dims.pitch * dims.height;
        }
    };

    template <typename T>
    struct PtrImpl<T3D, T> {
        FK_HOST_FUSE size_t sizeInBytes(const PtrDims<T3D>& dims) {
            return dims.color_planes_pitch * dims.color_planes;
        }
        FK_HOST_FUSE uint getNumElements(const PtrDims<T3D>& dims) {
            return dims.width * dims.height * dims.planes * dims.color_planes;
        }
        FK_HOST_FUSE void d_malloc(RawPtr<T3D, T>& ptr_a) {
            PtrImpl<T3D, T>::h_malloc_init(ptr_a.dims);
            gpuErrchk(cudaMalloc(&ptr_a.data, PtrImpl<T3D, T>::sizeInBytes(ptr_a.dims)));
        }
        FK_HOST_FUSE void h_malloc_init(PtrDims<T3D>& dims) {
            dims.pitch = sizeof(T) * dims.width;
            dims.plane_pitch = dims.pitch * dims.height;
            dims.color_planes_pitch = dims.plane_pitch * dims.planes;
        }
    };

    template <enum ND D, typename T>
    class Ptr {
        using Type = T;
        using At = PtrAccessor<D>;

    protected:
        struct RefPtr {
            void* ptr;
            int cnt;
        };
        RefPtr* ref{ nullptr };
        RawPtr<D, T> ptr_a;
        RawPtr<D, T> ptr_pinned;
        MemType type;
        int deviceID;

        inline constexpr Ptr(RefPtr* ref_, const RawPtr<D, T>& ptr_a_, const RawPtr<D, T>& ptr_pinned_, const MemType& type_, const int& devID) :
            ref(ref_), ptr_a(ptr_a_), ptr_pinned(ptr_pinned_), type(type_), deviceID(devID) {}
        inline constexpr Ptr(const RawPtr<D, T>& ptr_a_, RefPtr* ref_, const MemType& type_, const int& devID) :
            ref(ref_), ptr_a(ptr_a_), ptr_pinned(ptr_a_), type(type_), deviceID(devID) {}

        inline constexpr void allocDevice() {
            #if defined(__NVCC__) || defined(__HIP__)
            int currentDevice;
            gpuErrchk(cudaGetDevice(&currentDevice));
            gpuErrchk(cudaSetDevice(deviceID));
            PtrImpl<D, T>::d_malloc(ptr_a);
            if (currentDevice != deviceID) {
                gpuErrchk(cudaSetDevice(currentDevice));
            }
            #else
            throw std::runtime_error("Device allocation not supported in non-CUDA compilation.");
            #endif
        }

        inline constexpr void allocHost() {
            PtrImpl<D, T>::h_malloc_init(ptr_a.dims);
            ptr_a.data = (T*)malloc(PtrImpl<D, T>::sizeInBytes(ptr_a.dims));
        }

        inline constexpr void allocHostPinned() {
            #if defined(__NVCC__) || defined(__HIP__)
            int currentDevice;
            gpuErrchk(cudaGetDevice(&currentDevice));
            gpuErrchk(cudaSetDevice(deviceID));
            PtrImpl<D, T>::h_malloc_init(ptr_a.dims);
            gpuErrchk(cudaMallocHost(&ptr_a.data, PtrImpl<D, T>::sizeInBytes(ptr_a.dims)));
            if (currentDevice != deviceID) {
                gpuErrchk(cudaSetDevice(currentDevice));
            }
            #else
            throw std::runtime_error("Host pinned allocation not supported in non-CUDA compilation.");
            #endif
        }

        inline constexpr void allocDeviceAndPinned() {
            #if defined(__NVCC__) || defined(__HIP__)
            int currentDevice;
            gpuErrchk(cudaGetDevice(&currentDevice));
            gpuErrchk(cudaSetDevice(deviceID));
            PtrImpl<D, T>::d_malloc(ptr_a);
            PtrImpl<D, T>::h_malloc_init(ptr_pinned.dims);
            gpuErrchk(cudaMallocHost(&ptr_pinned.data, PtrImpl<D, T>::sizeInBytes(ptr_pinned.dims)));
            if (currentDevice != deviceID) {
                gpuErrchk(cudaSetDevice(currentDevice));
            }
            #else
            throw std::runtime_error("Host pinned and Device allocations not supported in non-CUDA compilation.");
            #endif
        }

        inline constexpr void freePrt() {
            if (ref) {
                ref->cnt--;
                if (ref->cnt == 0) {
                    switch (type) {
                    case Device:
                        {
                            #if defined(__NVCC__) || defined(__HIP__)
                            gpuErrchk(cudaFree(ref->ptr));
                            #else
                            throw std::runtime_error("Device memory deallocation not supported in non-CUDA compilation.");
                            #endif
                        }
                        break;
                    case Host:
                        free(ref->ptr);
                        break;
                    case HostPinned:
                        {
                            #if defined(__NVCC__) || defined(__HIP__)
                            gpuErrchk(cudaFreeHost(ref->ptr));
                            #else
                            throw std::runtime_error("Host pinned memory deallocation not supported in non-CUDA compilation.");
                            #endif
                        }
                        break;
                    case DeviceAndPinned:
                    {
#if defined(__NVCC__) || defined(__HIP__)
                        gpuErrchk(cudaFree(ref->ptr));
                        gpuErrchk(cudaFreeHost(ptr_pinned.data));
#else
                        throw std::runtime_error("Device and Host pinned memory deallocation not supported in non-CUDA compilation.");
#endif
                    }
                    break;
                    default:
                        break;
                    }
                    free(ref);
                }
            }
        }

        inline constexpr void initFromOther(const Ptr<D, T>& other) {
            ptr_a = other.ptr_a;
            ptr_pinned = other.ptr_pinned;
            type = other.type;
            deviceID = other.deviceID;
            if (other.ref) {
                ref = other.ref;
                ref->cnt++;
            }
        }
#if defined(__NVCC__) || defined(__HIP__)
        inline void copy(const RawPtr<D, T>& thisPtr, const RawPtr<D, T>& other, const cudaMemcpyKind& kind,
                         const cudaStream_t& stream = 0) {
            if ((other.dims.pitch == other.dims.width * sizeof(T)) && (thisPtr.dims.pitch == thisPtr.dims.width * sizeof(T))) {
                if (sizeInBytes() != PtrImpl<D, T>::sizeInBytes(other.dims)) {
                    throw std::runtime_error("Size mismatch in upload.");
                }
                const size_t totalBytes = sizeInBytes();
                gpuErrchk(cudaMemcpyAsync(other.data, thisPtr.data, totalBytes, kind, stream));
            } else {
                if constexpr (D > _2D || D == _1D) {
                    throw std::runtime_error("Padding only supported in 2D pointers");
                } else {
                    gpuErrchk(cudaMemcpy2DAsync(other.data, other.dims.pitch, thisPtr.data, thisPtr.dims.pitch,
                        thisPtr.dims.width * sizeof(T), thisPtr.dims.height, kind, stream));
                }
            }
        }
#endif
    public:

        inline constexpr Ptr() {}

        inline constexpr Ptr(const Ptr<D, T>& other) {
            initFromOther(other);
        }

        inline constexpr Ptr(const PtrDims<D>& dims, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            allocPtr(dims, type_, deviceID_);
        }

        inline constexpr Ptr(T* data_, const PtrDims<D>& dims, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            if (type_ == MemType::DeviceAndPinned) {
                throw std::runtime_error("DeviceAndPinned type requires an additional argument for the pinned pointer.");
            }
            ptr_a.data = data_;
            ptr_a.dims = dims;
            ptr_pinned = ptr_a;
            type = type_;
            deviceID = deviceID_;
        }

        inline constexpr Ptr(T* data_, T* pinned_data, const PtrDims<D>& dims, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            ptr_a.data = data_;
            ptr_a.dims = dims;
            ptr_pinned.data = pinned_data;
            ptr_pinned.dims = dims;
            type = type_;
            deviceID = deviceID_;
        }

        inline constexpr void allocPtr(const PtrDims<D>& dims_, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            ptr_a.dims = dims_;
            ptr_pinned.dims = dims_;
            type = type_;
            deviceID = deviceID_;
            ref = (RefPtr*)malloc(sizeof(RefPtr));
            if (ref != nullptr) {
                ref->cnt = 1;
            } else {
                throw std::runtime_error("Failed to allocate memory for reference counter.");
            }

            switch (type) {
            case Device:
                {
                    allocDevice();
                    ptr_pinned = ptr_a;
                }
                break;
            case Host:
                {
                    allocHost();
                    ptr_pinned = ptr_a;
                }
                break;
            case HostPinned:
                {
                    allocHostPinned();
                    ptr_pinned = ptr_a;
                }
                break;
            case DeviceAndPinned:
                {
                    allocDeviceAndPinned();
                }
                break;
            default:
                break;
            }

            ref->ptr = ptr_a.data;
        }

        inline ~Ptr() {
            freePrt();
        }

        inline constexpr RawPtr<D, T> ptr() const { return ptr_a; }
        inline constexpr RawPtr<D, T> ptrPinned() const { return ptr_pinned; }

        inline constexpr operator RawPtr<D, T>() const { return ptr_a; }

        inline constexpr Ptr<D, T> crop(const Point& p, const PtrDims<D>& newDims) {
            T* ptr = At::point(p, ptr_a);
            ref->cnt++;
            const RawPtr<D, T> newRawPtr = { ptr, newDims };
            if (type == MemType::DeviceAndPinned) {
                T* pinnedPtr = At::point(p, ptr_pinned);
                RawPtr<D, T> newPinnedRawPtr = { pinnedPtr, newDims };
                return { ref, newRawPtr, newPinnedRawPtr, type, deviceID };
            } else {
                return { ref, newRawPtr, newRawPtr, type, deviceID };
            }
        }

        inline constexpr PtrDims<D> dims() const {
            return ptr_a.dims;
        }

        inline constexpr MemType getMemType() const {
            return type;
        }

        inline constexpr int getDeviceID() const {
            return deviceID;
        }

        inline constexpr size_t sizeInBytes() const {
            return PtrImpl<D, T>::sizeInBytes(ptr_a.dims);
        }

        inline constexpr uint getNumElements() const {
            return PtrImpl<D, T>::getNumElements(ptr_a.dims);
        }

        inline constexpr int getRefCount() const {
            return ref->cnt;
        }

        Ptr<D, T>& operator=(const Ptr<D, T>& other) {
            initFromOther(other);
            return *this;
        }

#if defined(__NVCC__) || defined(__HIP__)
        inline void uploadTo(const Ptr<D, T>& other, const cudaStream_t& stream = 0) {
            constexpr cudaMemcpyKind kind = cudaMemcpyHostToDevice;
            constexpr MemType otherExpectedMemType1 = MemType::Device;
            constexpr MemType otherExpectedMemType2 = MemType::DeviceAndPinned;
            constexpr MemType thisExpectedMemType1 = MemType::Host;
            constexpr MemType thisExpectedMemType2 = MemType::HostPinned;
            if (type == thisExpectedMemType1 || type == thisExpectedMemType2) {
                if (other.getMemType() == otherExpectedMemType1 || other.getMemType() == otherExpectedMemType2) {
                    copy(ptr_a, other, kind, stream);
                } else {
                    throw std::runtime_error("Upload can only copy to Device pointers");
                }
            } else {
                throw std::runtime_error("Upload can only copy from Host or HostPinned pointers.");
            }
        }

        inline void downloadTo(const Ptr<D, T>& other, const cudaStream_t& stream = 0) {
            constexpr cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
            constexpr MemType otherExpectedMemType1 = MemType::Host;
            constexpr MemType otherExpectedMemType2 = MemType::HostPinned;
            constexpr MemType thisExpectedMemType1 = MemType::Device;
            constexpr MemType thisExpectedMemType2 = MemType::DeviceAndPinned;
            if (type == thisExpectedMemType1 || type == thisExpectedMemType2) {
                if (other.getMemType() == otherExpectedMemType1 || other.getMemType() == otherExpectedMemType2) {
                    copy(ptr_a, other, kind, stream);
                } else {
                    throw std::runtime_error("Download can only copy to Host or HostPinned pointers.");
                }
            } else {
                throw std::runtime_error("Download can only copy from Device pointers.");
            }
        }

        inline void upload(const Stream& stream) {
            if (type == MemType::DeviceAndPinned) {
                constexpr cudaMemcpyKind kind = cudaMemcpyHostToDevice;
                copy(ptr_pinned, ptr_a, kind, stream);
            }
        }
        inline void download(const Stream& stream) {
            if (type == MemType::DeviceAndPinned) {
                constexpr cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
                copy(ptr_a, ptr_pinned, kind, stream);
            }
        }
#else
        inline void upload(const Stream& stream) {}
        inline void download(const Stream& stream) {}
#endif

        inline T at(const Point& p) const {
            if (type != MemType::Device) {
                return *At::cr_point(p, ptr_pinned);
            } else {
                throw std::runtime_error("Cannot access data in Device memory from host code");
                return make_set<T>(0);
            }
        }

        inline T& at(const Point& p) {
            if (type != MemType::Device) {
                return *At::point(p, ptr_pinned);
            } else {
                throw std::runtime_error("Cannot access data in Device memory from host code");
                //return make_set<T>(0);
            }
        }

        template <enum ND DIM = D>
        inline std::enable_if_t<(DIM > _2D), Ptr<_2D, T>> getPlane(const uint& plane) {
            if (plane >= this->ptr_pinned.dims.planes) {
                throw std::runtime_error("Plane index out of bounds");
            }

            T* const data = PtrAccessor<_3D>::point(Point(0, 0, plane), this->ptr_a);
            T* const pinned_data = PtrAccessor<_3D>::point(Point(0, 0, plane), this->ptr_pinned);
            return Ptr<_2D, T>(data, pinned_data, PtrDims<_2D>{this->ptr_a.dims.width, this->ptr_a.dims.height, this->ptr_a.dims.pitch}, this->type, this->deviceID);
        }
    };

    template <typename T>
    class Ptr1D : public Ptr<_1D, T> {
    public:
        inline constexpr Ptr1D<T>() {}
        inline constexpr Ptr1D<T>(const uint& num_elems, const uint& size_in_bytes = 0, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<_1D, T>(PtrDims<_1D>(num_elems, size_in_bytes), type_, deviceID_) {}

        inline constexpr Ptr1D<T>(const Ptr<_1D, T>& other) : Ptr<_1D, T>(other) {}

        inline constexpr Ptr1D<T>(T* data_, const PtrDims<_1D>& dims_, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<_1D, T>(data_, dims_, type_, deviceID_) {}

        inline constexpr Ptr1D<T> crop1D(const Point& p, const PtrDims<_1D>& newDims) { return Ptr<_1D, T>::crop(p, newDims); }
    };

    template <typename T>
    class Ptr2D : public Ptr<_2D, T> {
    public:
        inline constexpr Ptr2D<T>() {}
        inline Ptr2D<T>(const Size& size, const uint& pitch_ = 0, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<_2D, T>(PtrDims<_2D>(size.width, size.height, pitch_), type_, deviceID_) {}
        inline Ptr2D<T>(const uint& width_, const uint& height_, const uint& pitch_ = 0, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<_2D, T>(PtrDims<_2D>(width_, height_, pitch_), type_, deviceID_) {}

        inline constexpr Ptr2D<T>(const Ptr<_2D, T>& other) : Ptr<_2D, T>(other) {}

        inline Ptr2D<T>(T* data_, const uint& width_, const uint& height_, const uint& pitch_, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<_2D, T>(data_, PtrDims<_2D>(width_, height_, pitch_), type_, deviceID_) {}

        inline Ptr2D<T> crop2D(const Point& p, const PtrDims<_2D>& newDims) { return Ptr<_2D, T>::crop(p, newDims); }
        inline void Alloc(const fk::Size& size, const uint& pitch_ = 0, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            this->allocPtr(PtrDims<_2D>(size.width, size.height, pitch_), type_, deviceID_);
        }
    };

    // A Ptr3D pointer
    template <typename T>
    class Ptr3D : public Ptr<_3D, T> {
    public:
        inline constexpr Ptr3D<T>(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const uint& pitch_ = 0, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<_3D, T>(PtrDims<_3D>(width_, height_, planes_, color_planes_, pitch_), type_, deviceID_) {}

        inline constexpr Ptr3D<T>(const Ptr<_3D, T>& other) : Ptr<_3D, T>(other) {}

        inline constexpr Ptr3D<T>(T* data_, const PtrDims<_3D>& dims_, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<_3D, T>(data_, dims_, type_, deviceID_) {}

        inline constexpr Ptr3D<T> crop3D(const Point& p, const PtrDims<_3D>& newDims) { return Ptr<_3D, T>::crop(p, newDims); }
    };

    // A color-plane-transposed 3D pointer PtrT3D
    template <typename T>
    class PtrT3D : public Ptr<T3D, T> {
    public:
        inline constexpr PtrT3D<T>(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<T3D, T>(PtrDims<T3D>(width_, height_, planes_, color_planes_), type_, deviceID_) {}

        inline constexpr PtrT3D<T>(const Ptr<T3D, T>& other) : Ptr<T3D, T>(other) {}

        inline constexpr PtrT3D<T>(T* data_, const PtrDims<T3D>& dims_, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<T3D, T>(data_, dims_, type_, deviceID_) {}

        inline constexpr PtrT3D<T> crop3D(const Point& p, const PtrDims<T3D>& newDims) { return Ptr<T3D, T>::crop(p, newDims); }
    };

    // A Tensor pointer
    template <typename T>
    class Tensor : public Ptr<_3D, T> {
    public:
        inline constexpr Tensor() {}

        inline constexpr Tensor(const Tensor<T>& other) : Ptr<_3D, T>(other) {}

        inline constexpr Tensor(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<_3D, T>(PtrDims<_3D>(width_, height_, planes_, color_planes_, sizeof(T)* width_), type_, deviceID_) {}

        inline constexpr Tensor(T* data, const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<_3D, T>(data, PtrDims<_3D>(width_, height_, planes_, color_planes_, sizeof(T)* width_), type_, deviceID_) {}

        inline constexpr void allocTensor(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            this->allocPtr(PtrDims<_3D>(width_, height_, planes_, color_planes_, sizeof(T) * width_), type_, deviceID_);
        }
    };

    // A color-plane-transposed Tensor pointer
    template <typename T>
    class TensorT : public Ptr<T3D, T> {
    public:
        inline constexpr TensorT() {}

        inline constexpr TensorT(const TensorT<T>& other) : Ptr<T3D, T>(other) {}

        inline constexpr TensorT(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<T3D, T>(PtrDims<T3D>(width_, height_, planes_, color_planes_), type_, deviceID_) {}

        inline constexpr TensorT(T* data, const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<T3D, T>(data, PtrDims<T3D>(width_, height_, planes_, color_planes_), type_, deviceID_) {}

        inline constexpr void allocTensor(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            this->allocPtr(PtrDims<T3D>(width_, height_, planes_, color_planes_), type_, deviceID_);
        }
    };

} // namespace fk

#endif
