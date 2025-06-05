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

#ifndef FK_VECTOR_TYPES
#define FK_VECTOR_TYPES

#include <fused_kernel/core/utils/type_lists.h>

using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;
using ulong = unsigned long int;
using ulonglong = unsigned long long int;

namespace fk {

#if defined(_MSC_VER) && _MSC_VER >= 1910 && _MSC_VER <= 1916
    struct Bool1 {
        bool x;
    };

    struct Bool2 {
        bool x, y;
    };

    struct Bool3 {
        bool x, y, z;
    };

    struct Bool4 {
        bool x, y, z, w;
    };
} // namespace fk

using bool1 = fk::Bool1;
using bool2 = fk::Bool2;
using bool3 = fk::Bool3;
using bool4 = fk::Bool4;

namespace fk {
    struct Char1 {
        signed char x;
    };

    struct Uchar1 {
        uchar x;
    };

    struct alignas(2) Char2 {
        struct { signed char x, y; };
    };

    union alignas(2) Uchar2 {
        struct { uchar x, y; };
    };

    struct Char3 {
        struct { signed char x, y, z; };
    };

    struct Uchar3 {
        struct { uchar x, y, z; };
    };

    struct alignas(4) Char4 {
        struct { signed char x, y, z, w; };
    };

    union alignas(4) Uchar4 {
        struct { uchar x, y, z, w; };
    };

    struct Short1 {
        short x;
    };

    struct Ushort1 {
        ushort x;
    };

    struct alignas(4) Short2 {
        struct { short x, y; };
    };

    struct alignas(4) Ushort2 {
        struct { ushort x, y; };
    };

    struct Short3 {
        struct { short x, y, z; };
    };

    struct Ushort3 {
        struct { ushort x, y, z; };
    };

    struct alignas(8) Short4 {
        struct { short x, y, z, w; };
    };

    struct alignas(8) Ushort4 {
        struct { ushort x, y, z, w; };
    };

    struct Int1 {
        int x;
    };

    struct Uint1 {
        unsigned int x;
    };

    struct alignas(8) Int2 {
        struct { int x, y; };
    };

    struct alignas(8) Uint2 {
        struct { unsigned int x, y; };
    };

    struct Int3 {
        struct { int x, y, z; };
    };

    struct Uint3 {
        struct { unsigned int x, y, z; };
    };

    struct alignas(16) Int4 {
        struct { int x, y, z, w; };
    };

    struct alignas(16) Uint4 {
        struct { unsigned int x, y, z, w; };
    };

    struct Long1 {
        long int x;
    };

    struct Ulong1 {
        ulong x;
    };

    struct alignas(2 * sizeof(long int)) Long2 {
        struct { long int x, y; };
    };

    struct alignas(2 * sizeof(unsigned long int)) Ulong2 {
        struct { ulong x, y; };
    };

    struct Long3 {
        struct { long int x, y, z; };
    };

    struct Ulong3 {
        struct { ulong x, y, z; };
    };

    union alignas(16) Long4 {
        struct { long int x, y, z, w; };
    };

    struct alignas(16) Ulong4 {
        struct { ulong x, y, z, w; };
    };

    struct Float1 {
        float x;
    };

    struct alignas(8) Float2 {
        struct { float x, y; };
    };

    struct Float3 {
        struct { float x, y, z; };
    };

    struct alignas(16) Float4 {
        struct { float x, y, z, w; };
    };

    struct Longlong1 {
        long long int x;
    };

    struct Ulonglong1 {
        ulonglong x;
    };

    struct alignas(16) Longlong2 {
        struct { long long int x, y; };
    };

    struct alignas(16) Ulonglong2 {
        struct { ulonglong x, y; };
    };

    struct Longlong3 {
        struct { long long int x, y, z; };
    };

    struct Ulonglong3 {
        struct { ulonglong x, y, z; };
    };

    struct alignas(16) Longlong4 {
        struct { long long int x, y, z, w; };
    };

    struct alignas(16) Ulonglong4 {
        struct { ulonglong x, y, z, w; };
    };

    struct Double1 {
        double x;
    };

    struct alignas(16) Double2 {
        struct { double x, y; };
    };

    struct alignas(16) Double3 {
        struct { double x, y, z; };
    };

    struct alignas(16) Double4 {
        struct { double x, y, z, w; };
    };
} // namespace fk

#else
    union Bool1 {
        bool x;
        bool at[1];
    };

    union Bool2 {
        struct { bool x, y; };
        bool at[2];
    };

    union Bool3 {
        struct { bool x, y, z; };
        bool at[3];
    };

    union Bool4 {
        struct { bool x, y, z, w; };
        bool at[4];
    };
} // namespace fk

using bool1 = fk::Bool1;
using bool2 = fk::Bool2;
using bool3 = fk::Bool3;
using bool4 = fk::Bool4;

namespace fk {
    union Char1 {
        signed char x;
        signed char at[1];
    };

    union Uchar1 {
        uchar x;
        uchar at[1];
    };

    union alignas(2) Char2 {
        struct { signed char x, y; };
        signed char at[2];
    };

    union alignas(2) Uchar2 {
        struct { uchar x, y; };
        uchar at[2];
    };

    union Char3 {
        struct { signed char x, y, z; };
        signed char at[3];
    };

    union Uchar3 {
        struct { uchar x, y, z; };
        uchar at[3];
    };

    union alignas(4) Char4 {
        struct { signed char x, y, z, w; };
        signed char at[4];
    };

    union alignas(4) Uchar4 {
        struct { uchar x, y, z, w; };
        uchar at[4];
    };

    union Short1 {
        short x;
        short at[1];
    };

    union Ushort1 {
        ushort x;
        ushort at[1];
    };

    union alignas(4) Short2 {
        struct { short x, y; };
        short at[2];
    };

    union alignas(4) Ushort2 {
        struct { ushort x, y; };
        ushort at[2];
    };

    union Short3 {
        struct { short x, y, z; };
        short at[3];
    };

    union Ushort3 {
        struct { ushort x, y, z; };
        ushort at[3];
    };

    union alignas(8) Short4 {
        struct { short x, y, z, w; };
        short at[4];
    };

    union alignas(8) Ushort4 {
        struct { ushort x, y, z, w; };
        ushort at[4];
    };

    union Int1 {
        int x;
        int at[1];
    };

    union Uint1 {
        unsigned int x;
        unsigned int at[1];
    };

    union alignas(8) Int2 {
        struct { int x, y; };
        int at[2];
    };

    union alignas(8) Uint2 {
        struct { unsigned int x, y; };
        unsigned int at[2];
    };

    union Int3 {
        struct { int x, y, z; };
        int at[3];
    };

    union Uint3 {
        struct { unsigned int x, y, z; };
        unsigned int at[3];
    };

    union alignas(16) Int4 {
        struct { int x, y, z, w; };
        int at[4];
    };

    union alignas(16) Uint4 {
        struct { unsigned int x, y, z, w; };
        unsigned int at[4];
    };

    union Long1 {
        long int x;
        long int at[1];
    };

    union Ulong1 {
        ulong x;
        ulong at[1];
    };

    union alignas(2 * sizeof(long int)) Long2 {
        struct { long int x, y; };
        long int at[2];
    };

    union alignas(2 * sizeof(unsigned long int)) Ulong2 {
        struct { ulong x, y; };
        ulong at[2];
    };

    union Long3 {
        struct { long int x, y, z; };
        long int at[3];
    };

    union Ulong3 {
        struct { ulong x, y, z; };
        ulong at[3];
    };

    union alignas(16) Long4 {
        struct { long int x, y, z, w; };
        long int at[4];
    };

    union alignas(16) Ulong4 {
        struct { ulong x, y, z, w; };
        ulong at[4];
    };

    union Float1 {
        float x;
        float at[1];
    };

    union alignas(8) Float2 {
        struct { float x, y; };
        float at[2];
    };

    union Float3 {
        struct { float x, y, z; };
        float at[3];
    };

    union alignas(16) Float4 {
        struct { float x, y, z, w; };
        float at[4];
    };

    union Longlong1 {
        long long int x;
        long long int at[1];
    };

    union Ulonglong1 {
        ulonglong x;
        ulonglong at[1];
    };

    union alignas(16) Longlong2 {
        struct { long long int x, y; };
        long long int at[2];
    };

    union alignas(16) Ulonglong2 {
        struct { ulonglong x, y; };
        ulonglong at[2];
    };

    union Longlong3 {
        struct { long long int x, y, z; };
        long long int at[3];
    };

    union Ulonglong3 {
        struct { ulonglong x, y, z; };
        ulonglong at[3];
    };

    union alignas(16) Longlong4 {
        struct { long long int x, y, z, w; };
        long long int at[4];
    };

    union alignas(16) Ulonglong4 {
        struct { ulonglong x, y, z, w; };
        ulonglong at[4];
    };

    union Double1 {
        double x;
        double at[1];
    };

    union alignas(16) Double2 {
        struct { double x, y; };
        double at[2];
    };

    union alignas(16) Double3 {
        struct { double x, y, z; };
        double at[3];
    };

    union alignas(16) Double4 {
        struct { double x, y, z, w; };
        double at[4];
    };
} // namespace fk
#endif // 

#if defined(__NVCC__) || defined(__HIP__)
#include <vector_types.h>
#else
using char1 = fk::Char1;
using uchar1 = fk::Uchar1;
using char2 = fk::Char2;
using uchar2 = fk::Uchar2;
using char3 = fk::Char3;
using uchar3 = fk::Uchar3;
using char4 = fk::Char4;
using uchar4 = fk::Uchar4;
using short1 = fk::Short1;
using ushort1 = fk::Ushort1;
using short2 = fk::Short2;
using ushort2 = fk::Ushort2;
using short3 = fk::Short3;
using ushort3 = fk::Ushort3;
using short4 = fk::Short4;
using ushort4 = fk::Ushort4;
using int1 = fk::Int1;
using uint1 = fk::Uint1;
using int2 = fk::Int2;
using uint2 = fk::Uint2;
using int3 = fk::Int3;
using uint3 = fk::Uint3;
using int4 = fk::Int4;
using uint4 = fk::Uint4;
using long1 = fk::Long1;
using ulong1 = fk::Ulong1;
using long2 = fk::Long2;
using ulong2 = fk::Ulong2;
using long3 = fk::Long3;
using ulong3 = fk::Ulong3;
using long4 = fk::Long4;
using ulong4 = fk::Ulong4;
using float1 = fk::Float1;
using float2 = fk::Float2;
using float3 = fk::Float3;
using float4 = fk::Float4;
using longlong1 = fk::Longlong1;
using ulonglong1 = fk::Ulonglong1;
using longlong2 = fk::Longlong2;
using ulonglong2 = fk::Ulonglong2;
using longlong3 = fk::Longlong3;
using ulonglong3 = fk::Ulonglong3;
using longlong4 = fk::Longlong4;
using ulonglong4 = fk::Ulonglong4;
using double1 = fk::Double1;
using double2 = fk::Double2;
using double3 = fk::Double3;
using double4 = fk::Double4;
#endif /* __NVCC__ || __HIPCC__ */

#endif /* FK_VECTOR_TYPES */