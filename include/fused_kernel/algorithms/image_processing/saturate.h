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

#ifndef FK_SATURATE
#define FK_SATURATE

#include <cmath>

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/core/utils/vlimits.h>

namespace fk {

    template <typename I, typename O>
    struct SaturateCastBase;

    #define SATURATE_CAST_BASE(IT) \
    template <typename O> \
    struct SaturateCastBase<IT, O> { \
    private: \
        using SelfType = SaturateCastBase<IT, O>; \
    public: \
        FK_STATIC_STRUCT_SELFTYPE(SaturateCastBase, SelfType) \
        using InputType = IT; \
        using OutputType = O; \
        using InstanceType = UnaryType; \
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) { \
            return static_cast<O>(input); \
        } \
    };

    SATURATE_CAST_BASE(uchar)
    SATURATE_CAST_BASE(char)
    SATURATE_CAST_BASE(schar)
    SATURATE_CAST_BASE(ushort)
    SATURATE_CAST_BASE(short)
    SATURATE_CAST_BASE(uint)
    SATURATE_CAST_BASE(int)
    SATURATE_CAST_BASE(float)
    SATURATE_CAST_BASE(double)

    #undef SATURATE_CAST_BASE

    #define SATURATE_CAST_BASE(IT, OT) \
    template <> \
    struct SaturateCastBase<IT, OT> { \
    private: \
        using SelfType = SaturateCastBase<IT, OT>; \
    public: \
        FK_STATIC_STRUCT_SELFTYPE(SaturateCastBase, SelfType) \
        using InputType = IT; \
        using OutputType = OT; \
        using InstanceType = UnaryType;

    SATURATE_CAST_BASE(double, float)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
#ifdef __CUDA_ARCH__
            return __double2float_rn(input);
#else
            if (input > maxValue<OutputType>) {
                return maxValue<OutputType>;
            } else if (input < fk::minValue<OutputType>) {
                return minValue<OutputType>;
            } else {
                return static_cast<OutputType>(input);
            }
#endif
        }
    };

    SATURATE_CAST_BASE(schar, uchar)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            const int vi = static_cast<int>(input);
            if (vi < 0) {
                return 0;
            } else if (vi > 255) {
                return 255;
            } else {
                return static_cast<uchar>(vi);
            }
        }
    };

    SATURATE_CAST_BASE(char, uchar)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            const int vi = static_cast<int>(input);
            if (vi < 0) {
                return 0;
            } else if (vi > 255) {
                return 255;
            } else {
                return static_cast<uchar>(vi);
            }
        }
    };

    SATURATE_CAST_BASE(short, uchar)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            const int vi = static_cast<int>(input);
            if (vi < 0) {
                return 0;
            } else if (vi > 255) {
                return 255;
            } else {
                return static_cast<uchar>(vi);
            }
        }
    };

    SATURATE_CAST_BASE(ushort, uchar)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            if (input > 255) {
                return 255;
            } else {
                return static_cast<uchar>(input);
            }
        }
    };

    SATURATE_CAST_BASE(int, uchar)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            if (input < 0) {
                return 0;
            } else if (input > 255) {
                return 255;
            } else {
                return static_cast<uchar>(input);
            }
        }
    };
    SATURATE_CAST_BASE(uint, uchar)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            if (input > 255) {
                return 255;
            } else {
                return static_cast<uchar>(input);
            }
        }
    };
    SATURATE_CAST_BASE(float, uchar)
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
            const int vi = __float2uint_rn(input);
            if (vi > 255) {
                return 255;
            } else {
                return static_cast<uchar>(vi);
            }
    #else
            const int vi = static_cast<int>(std::nearbyint(input));
            if (vi < 0) {
                return 0;
            } else if (vi > 255) {
                return 255;
            } else {
                return static_cast<uchar>(vi);
            }
    #endif
        }
    };
    SATURATE_CAST_BASE(double, uchar)
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
            const uint vi = __double2uint_rn(input);
            if (vi > 255) {
                return 255;
            } else {
                return static_cast<uchar>(vi);
            }
    #else
            const int vi = static_cast<int>(std::nearbyint(input));
            if (vi < 0) {
                return 0;
            } else if (vi > 255) {
                return 255;
            } else {
                return static_cast<uchar>(vi);
            }
    #endif
        }
    };
    SATURATE_CAST_BASE(uchar, schar)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            if (input > 127) {
                return 127;
            } else {
                return static_cast<OutputType>(input);
            }
        }
    };
    SATURATE_CAST_BASE(uchar, char)
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
        if (input > 127) {
            return 127;
        } else {
            return static_cast<OutputType>(input);
        }
    }
    };
    SATURATE_CAST_BASE(short, schar)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            if (input < -128) {
                return -128;
            } else if (input > 127) {
                return 127;
            } else {
                return static_cast<OutputType>(input);
            }
        }
    };
    SATURATE_CAST_BASE(short, char)
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
        if (input < -128) {
            return -128;
        } else if (input > 127) {
            return 127;
        } else {
            return static_cast<OutputType>(input);
        }
    }
    };
    SATURATE_CAST_BASE(ushort, schar)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            if (input > 127) {
                return 127;
            } else {
                return static_cast<OutputType>(input);
            }
        }
    };
    SATURATE_CAST_BASE(ushort, char)
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
        if (input > 127) {
            return 127;
        } else {
            return static_cast<OutputType>(input);
        }
    }
    };
    SATURATE_CAST_BASE(int, schar)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            if (input < -128) {
                return -128;
            } else if (input > 127) {
                return 127;
            } else {
                return static_cast<OutputType>(input);
            }
        }
    };
    SATURATE_CAST_BASE(int, char)
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
        if (input < -128) {
            return -128;
        } else if (input > 127) {
            return 127;
        } else {
            return static_cast<OutputType>(input);
        }
    }
    };
    SATURATE_CAST_BASE(uint, schar)
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
        if (input > 127) {
            return 127;
        } else {
            return static_cast<OutputType>(input);
        }
    }
    };
    SATURATE_CAST_BASE(uint, char)
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
        if (input > 127) {
            return 127;
    } else {
            return static_cast<OutputType>(input);
        }
        }
    };
    SATURATE_CAST_BASE(float, schar)
    FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
            const int vi = __float2int_rn(input);
    #else
            const int vi = static_cast<int>(std::nearbyint(input));
    #endif
            if (vi < -128) {
                return -128;
            } else if (vi > 127) {
                return 127;
            } else {
                return static_cast<OutputType>(vi);
            }
        }
    };
    SATURATE_CAST_BASE(float, char)
    FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
        const int vi = __float2int_rn(input);
    #else
        const int vi = static_cast<int>(std::nearbyint(input));
    #endif
        if (vi < -128) {
            return -128;
        } else if (vi > 127) {
            return 127;
        } else {
            return static_cast<OutputType>(vi);
        }
    }
    };
    SATURATE_CAST_BASE(schar, ushort)
    FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
            return __float2uint_rn(static_cast<float>(input));
    #else
            if (input < 0) {
                return 0;
            } else {
                return static_cast<OutputType>(input);
            }
    #endif
        }
    };
    SATURATE_CAST_BASE(char, ushort)
    FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
        return __float2uint_rn(static_cast<float>(input));
    #else
        if (input < 0) {
            return 0;
        } else {
            return static_cast<OutputType>(input);
        }
    #endif
    }
    };
    SATURATE_CAST_BASE(short, ushort)
    FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
            return __float2uint_rn(static_cast<float>(input));
    #else
            if (input < 0) {
                return 0;
            } else {
                return static_cast<OutputType>(input);
            }
    #endif
        }
    };
    SATURATE_CAST_BASE(int, ushort)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            if (input < 0) {
                return 0;
            } else if (input > 65535) {
                return 65535;
            } else {
                return static_cast<OutputType>(input);
            }
        }
    };
    SATURATE_CAST_BASE(uint, ushort)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            if (input > 65535) {
                return 65535;
            } else {
                return static_cast<OutputType>(input);
            }
        }
    };
    SATURATE_CAST_BASE(float, ushort)
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
            const int vi = __float2uint_rn(input);
            if (vi > 65535) {
                return 65535;
            } else {
                return static_cast<OutputType>(vi);
            }
    #else
            const int vi = static_cast<int>(std::nearbyint(input));
            if (vi < 0) {
                return 0;
            } else if (vi > 65535) {
                return 65535;
            } else {
                return static_cast<OutputType>(vi);
            }
    #endif
        }
    };
    SATURATE_CAST_BASE(double, ushort)
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
            const int vi = __double2uint_rn(input);
            if (vi > 65535) {
                return 65535;
            } else {
                return static_cast<OutputType>(vi);
            }
    #else
            const int vi = static_cast<int>(std::nearbyint(input));
            if (vi < 0) {
                return 0;
            } else if (vi > 65535) {
                return 65535;
            } else {
                return static_cast<OutputType>(vi);
            }
    #endif
        }
    };
    SATURATE_CAST_BASE(ushort, short)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            if (input > 32767) {
                return 32767;
            } else {
                return static_cast<OutputType>(input);
            }
        }
    };
    SATURATE_CAST_BASE(int, short)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            if (input < -32768) {
                return -32768;
            } else if (input > 32767) {
                return 32767;
            } else {
                return static_cast<OutputType>(input);
            }
        }
    };
    SATURATE_CAST_BASE(uint, short)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            if (input > 32767) {
                return 32767;
            } else {
                return static_cast<OutputType>(input);
            }
        }
    };
    SATURATE_CAST_BASE(float, short)
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
            const int vi = __float2int_rn(input);
    #else
            const int vi = static_cast<int>(std::nearbyint(input));
    #endif
            if (vi < -32768) {
                return -32768;
            } else if (vi > 32767) {
                return 32767;
            } else {
                return static_cast<OutputType>(vi);
            }
        }
    };
    SATURATE_CAST_BASE(double, short)
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
            const int vi = __double2int_rn(input);
    #else
            const int vi = static_cast<int>(std::nearbyint(input));
    #endif
            if (vi < -32768) {
                return -32768;
            } else if (vi > 32767) {
                return 32767;
            } else {
                return static_cast<OutputType>(vi);
            }
        }
    };
    SATURATE_CAST_BASE(uint, int)
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            if (input > 2147483647) {
                return 2147483647;
            } else {
                return static_cast<OutputType>(input);
            }
        }
    };
    SATURATE_CAST_BASE(float, int)
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
            return __float2int_rn(input);
    #else
            return static_cast<OutputType>(std::nearbyint(input));
    #endif
        }
    };
    SATURATE_CAST_BASE(double, int)
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
            return __double2int_rn(input);
    #else
            return static_cast<OutputType>(std::nearbyint(input));
    #endif
        }
    };
    SATURATE_CAST_BASE(schar, uint)
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
            return __float2uint_rn(static_cast<float>(input));
    #else
            if (input < 0) {
                return 0;
            } else {
                return static_cast<OutputType>(input);
            }
    #endif
        }
    };
    SATURATE_CAST_BASE(char, uint)
    FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
        return __float2uint_rn(static_cast<float>(input));
    #else
        if (input < 0) {
            return 0;
        } else {
            return static_cast<OutputType>(input);
        }
    #endif
    }
    };
    SATURATE_CAST_BASE(short, uint)
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
            return __float2uint_rn(static_cast<float>(input));
    #else
            if (input < 0) {
                return 0;
            } else {
                return static_cast<OutputType>(input);
            }
    #endif
        }
    };
    SATURATE_CAST_BASE(int, uint)
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
            return __float2uint_rn(static_cast<float>(input));
    #else
            if (input < 0) {
                return 0;
            } else {
                return static_cast<OutputType>(input);
            }
    #endif
        }
    };
    SATURATE_CAST_BASE(float, uint)
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
            return __float2uint_rn(input);
    #else
            return static_cast<OutputType>(std::nearbyint(input));
    #endif
        }
    };
    SATURATE_CAST_BASE(double, uint)
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
    #ifdef __CUDA_ARCH__
            return __double2uint_rn(input);
    #else
            return static_cast<OutputType>(std::nearbyint(input));
    #endif
        }
    };

    template <typename O> 
    struct SaturateCastBase<ulong, O> {
    private: 
        using SelfType = SaturateCastBase<ulong, O>;
        using Parent = UnaryOperation<ulong, O, SelfType>;
    public: 
        FK_STATIC_STRUCT_SELFTYPE(SaturateCastBase, SelfType)
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
            if (input > maxValue<O>) {
                return maxValue<O>;
            } else {
                return static_cast<OutputType>(input);
            }
        }
    };

    template <typename O>
    struct SaturateCastBase<long, O> {
    private:
        using SelfType = SaturateCastBase<long, O>;
        using Parent = UnaryOperation<long, O, SelfType>;
    public:
        FK_STATIC_STRUCT_SELFTYPE(SaturateCastBase, SelfType)
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
            if (input > maxValue<OutputType>) {
                return maxValue<OutputType>;
            } else if (input < minValue<O>) {
                return minValue<OutputType>;
            } else {
                return static_cast<OutputType>(input);
            }
        }
    };

    template <typename O>
    struct SaturateCastBase<ulonglong, O> {
    private:
        using SelfType = SaturateCastBase<ulonglong, O>;
        using Parent = UnaryOperation<ulonglong, O, SelfType>;
    public:
        FK_STATIC_STRUCT_SELFTYPE(SaturateCastBase, SelfType)
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
            if (input > maxValue<OutputType>) {
                return maxValue<OutputType>;
            } else {
                return static_cast<OutputType>(input);
            }
        }
    };

    template <typename O>
    struct SaturateCastBase<longlong, O> {
    private:
        using SelfType = SaturateCastBase<longlong, O>;
        using Parent = UnaryOperation<longlong, O, SelfType>;
    public:
        FK_STATIC_STRUCT_SELFTYPE(SaturateCastBase, SelfType)
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input) {
            if (input > maxValue<OutputType>) {
                return maxValue<OutputType>;
            } else if (input < minValue<OutputType>) {
                return minValue<OutputType>;
            } else {
                return static_cast<OutputType>(input);
            }
        }
    };

    #undef SATURATE_CAST_BASE

    template <typename I, typename O>
    struct SaturateCast {
    private:
        using SelfType = SaturateCast<I, O>;
    public:
        FK_STATIC_STRUCT_SELFTYPE(SaturateCast, SelfType)
        using Parent = UnaryOperation<I, O, SaturateCast<I, O>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return UnaryV<SaturateCastBase<VBase<I>, VBase<O>>, I, O>::exec(input);
        }
    };

    struct SaturateFloatBase {
        FK_STATIC_STRUCT_SELFTYPE(SaturateFloatBase, SaturateFloatBase)
        using Parent = UnaryOperation<float, float, SaturateFloatBase>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return Max<float, float, float, UnaryType>::exec({ 0.f, Min<float,float,float,UnaryType>::exec({ input, 1.f }) });
        }
    };

    template <typename T>
    struct Saturate {
    private:
        using SelfType = Saturate<T>;
    public:
        FK_STATIC_STRUCT_SELFTYPE(Saturate, SelfType)
        using Parent = BinaryOperation<T, VectorType_t<VBase<T>, 2>, T, Saturate<T>>;
        DECLARE_BINARY_PARENT
        using Base = typename VectorTraits<T>::base;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            static_assert(!validCUDAVec<T>, "Saturate only works with non cuda vector types");
            return Max<Base>::exec(Min<Base>::exec(input, { params.y }), { params.x });
        }
    };

    template <typename T>
    struct SaturateFloat {
    private:
        using SelfType = SaturateFloat<T>;
    public:
        FK_STATIC_STRUCT_SELFTYPE(SaturateFloat, SelfType)
        using Parent = UnaryOperation<T, T, SaturateFloat<T>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(std::is_same_v<VBase<T>, float>, "Saturate float only works with float base types.");
            return UnaryV<SaturateFloatBase, T, T>::exec(input);
        }
    };

} // namespace fk

#endif
