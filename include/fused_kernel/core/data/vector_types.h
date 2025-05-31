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
union Bool1 {
    bool at[1];
    bool x;
};

union Bool2 {
    bool at[2];
    struct { bool x, y; };
};

union Bool3 {
    bool at[3];
    struct { bool x, y, z; };
};

union Bool4 {
    bool at[4];
    struct { bool x, y, z, w; };
};
} // namespace fk

using bool1 = fk::Bool1;
using bool2 = fk::Bool2;
using bool3 = fk::Bool3;
using bool4 = fk::Bool4;

#define FK_VECTOR_2 alignas(2)
#define FK_VECTOR_4 alignas(4)
#define FK_VECTOR_8 alignas(8)
#define FK_VECTOR_16 alignas(16)
#define FK_VECTOR_(value) alignas(value)

namespace fk {
union Char1 {
    signed char at[1];
    signed char x;
};

union Uchar1 {
    uchar at[1];
    uchar x;
};

union FK_VECTOR_2 Char2 {
    signed char at[2];
    struct { signed char x, y; };
};

union FK_VECTOR_2 Uchar2 {
    uchar at[2];
    struct { uchar x, y; };
};

union Char3 {
    signed char at[3];
    struct { signed char x, y, z; };
};

union Uchar3 {
    uchar at[3];
    struct { uchar x, y, z; };
};

union FK_VECTOR_4 Char4 {
    signed char at[4];
    struct { signed char x, y, z, w; };
};

union FK_VECTOR_4 Uchar4 {
    uchar at[4];
    struct { uchar x, y, z, w; };
};

union Short1 {
    short at[1];
    short x;
};

union Ushort1 {
    ushort at[1];
    ushort x;
};

union FK_VECTOR_4 Short2 {
    short at[2];
    struct { short x, y; };
};

union FK_VECTOR_4 Ushort2 {
    ushort at[2];
    struct { ushort x, y; };
};

union Short3 {
    short at[3];
    struct { short x, y, z; };
};

union Ushort3 {
    ushort at[3];
    struct { ushort x, y, z; };
};

union FK_VECTOR_8 Short4 {
    short at[4];
    struct { short x, y, z, w; };
};

union FK_VECTOR_8 Ushort4 {
    ushort at[4];
    struct { ushort x, y, z, w; };
};

union Int1 {
    int at[1];
    int x;
};

union Uint1 {
    unsigned int at[1];
    unsigned int x;
};

union FK_VECTOR_8 Int2 {
    int at[2];
    struct { int x, y; };
};

union FK_VECTOR_8 Uint2 {
    unsigned int at[2];
    struct { unsigned int x, y; };
};

union Int3 {
    int at[3];
    struct { int x, y, z; };
};

union Uint3 {
    unsigned int at[3];
    struct { unsigned int x, y, z; };
};

union FK_VECTOR_16 Int4 {
    int at[4];
    struct { int x, y, z, w; };
};

union FK_VECTOR_16 Uint4 {
    unsigned int at[4];
    struct { unsigned int x, y, z, w; };
};

union Long1 {
    long int at[1];
    long int x;
};

union Ulong1 {
    ulong at[1];
    ulong x;
};

union FK_VECTOR_(2 * sizeof(long int)) Long2 {
    long int at[2];
    struct { long int x, y; };
};

union FK_VECTOR_(2 * sizeof(unsigned long int)) Ulong2 {
    ulong at[2];
    struct { ulong x, y; };
};

union Long3 {
    long int at[3];
    struct { long int x, y, z; };
};

union Ulong3 {
    ulong at[3];
    struct { ulong x, y, z; };
};

union FK_VECTOR_16 Long4 {
    long int at[4];
    struct { long int x, y, z, w; };
};

union FK_VECTOR_16 Ulong4 {
    ulong at[4];
    struct { ulong x, y, z, w; };
};

union Float1 {
    float at[1];
    float x;
};

union FK_VECTOR_8 Float2 {
    float at[2];
    struct { float x, y; };
};

union Float3 {
    float at[3];
    struct { float x, y, z; };
};

union FK_VECTOR_16 Float4 {
    float at[4];
    struct { float x, y, z, w; };
};

union Longlong1 {
    long long int at[1];
    long long int x;
};

union Ulonglong1 {
    ulonglong at[1];
    ulonglong x;
};

union FK_VECTOR_16 Longlong2 {
    long long int at[2];
    struct { long long int x, y; };
};

union FK_VECTOR_16 Ulonglong2 {
    ulonglong at[2];
    struct { ulonglong x, y; };
};

union Longlong3 {
    long long int at[3];
    struct { long long int x, y, z; };
};

union Ulonglong3 {
    ulonglong at[3];
    struct { ulonglong x, y, z; };
};

union FK_VECTOR_16 Longlong4 {
    long long int at[4];
    struct { long long int x, y, z, w; };
};

union FK_VECTOR_16 Ulonglong4 {
    ulonglong at[4];
    struct { ulonglong x, y, z, w; };
};

union Double1 {
    double at[1];
    double x;
};

union FK_VECTOR_16 Double2 {
    double at[2];
    struct { double x, y; };
};

union FK_VECTOR_16 Double3 {
    double at[3];
    struct { double x, y, z; };
};

union FK_VECTOR_16 Double4 {
    double at[4];
    struct { double x, y, z, w; };
};
} // namespace fk

#undef FK_VECTOR
#undef FK_VECTOR_2
#undef FK_VECTOR_4
#undef FK_VECTOR_8
#undef FK_VECTOR_16

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