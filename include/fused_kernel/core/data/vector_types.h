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

using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;
using ulong = unsigned long int;
using ulonglong = unsigned long long int;

union bool1 {
    bool at[1];
    bool x;
};

union bool2 {
    bool at[2];
    struct { bool x, y; };
};

union bool3 {
    bool at[3];
    struct { bool x, y, z; };
};

union bool4 {
    bool at[4];
    struct { bool x, y, z, w; };
};

#if defined(__NVCC__) || defined(__HIPCC__)
#include <vector_types.h>
#else
#define FK_VECTOR_2 alignas(2)
#define FK_VECTOR_4 alignas(4)
#define FK_VECTOR_8 alignas(8)
#define FK_VECTOR_16 alignas(16)
#define FK_VECTOR_(value) alignas(value)

union char1 {
    signed char at[1];
    signed char x;
};

union uchar1 {
    uchar at[1];
    uchar x;
};

union FK_VECTOR_2 char2 {
    signed char at[2];
    struct { signed char x, y; };
};

union FK_VECTOR_2 uchar2 {
    uchar at[2];
    struct { uchar x, y; };
};

union char3 {
    signed char at[3];
    struct { signed char x, y, z; };
};

union uchar3 {
    uchar at[3];
    struct { uchar x, y, z; };
};

union FK_VECTOR_4 char4 {
    signed char at[4];
    struct { signed char x, y, z, w; };
};

union FK_VECTOR_4 uchar4 {
    uchar at[4];
    struct { uchar x, y, z, w; };
};

union short1 {
    short at[1];
    short x;
};

union ushort1 {
    ushort at[1];
    ushort x;
};

union FK_VECTOR_4 short2 {
    short at[2];
    struct { short x, y; };
};

union FK_VECTOR_4 ushort2 {
    ushort at[2];
    struct { ushort x, y; };
};

union short3 {
    short at[3];
    struct { short x, y, z; };
};

union ushort3 {
    ushort at[3];
    struct { ushort x, y, z; };
};

union FK_VECTOR_8 short4 {
    short at[4];
    struct { short x, y, z, w; };
};

union FK_VECTOR_8 ushort4 {
    ushort at[4];
    struct { ushort x, y, z, w; };
};

union int1 {
    int at[1];
    int x;
};

union uint1 {
    unsigned int at[1];
    unsigned int x;
};

union FK_VECTOR_8 int2 {
    int at[2];
    struct { int x, y; };
};

union FK_VECTOR_8 uint2 {
    unsigned int at[2];
    struct { unsigned int x, y; };
};

union int3 {
    int at[3];
    struct { int x, y, z; };
};

union uint3 {
    unsigned int at[3];
    struct { unsigned int x, y, z; };
};

union FK_VECTOR_16 int4 {
    int at[4];
    struct { int x, y, z, w; };
};

union FK_VECTOR_16 uint4 {
    unsigned int at[4];
    struct { unsigned int x, y, z, w; };
};

union long1 {
    long int at[1];
    long int x;
};

union ulong1 {
    ulong at[1];
    ulong x;
};

union FK_VECTOR_(2 * sizeof(long int)) long2 {
    long int at[2];
    struct { long int x, y; };
};

union FK_VECTOR_(2 * sizeof(unsigned long int)) ulong2 {
    ulong at[2];
    struct { ulong x, y; };
};

union long3 {
    long int at[3];
    struct { long int x, y, z; };
};

union ulong3 {
    ulong at[3];
    struct { ulong x, y, z; };
};

union FK_VECTOR_16 long4 {
    long int at[4];
    struct { long int x, y, z, w; };
};

union FK_VECTOR_16 ulong4 {
    ulong at[4];
    struct { ulong x, y, z, w; };
};

union float1 {
    float at[1];
    float x;
};

union FK_VECTOR_8 float2 {
    float at[2];
    struct { float x, y; };
};

union float3 {
    float at[3];
    struct { float x, y, z; };
};

union FK_VECTOR_16 float4 {
    float at[4];
    struct { float x, y, z, w; };
};

union longlong1 {
    long long int at[1];
    long long int x;
};

union ulonglong1 {
    ulonglong at[1];
    ulonglong x;
};

union FK_VECTOR_16 longlong2 {
    long long int at[2];
    struct { long long int x, y; };
};

union FK_VECTOR_16 ulonglong2 {
    ulonglong at[2];
    struct { ulonglong x, y; };
};

union longlong3 {
    long long int at[3];
    struct { long long int x, y, z; };
};

union ulonglong3 {
    ulonglong at[3];
    struct { ulonglong x, y, z; };
};

union FK_VECTOR_16 longlong4 {
    long long int at[4];
    struct { long long int x, y, z, w; };
};

union FK_VECTOR_16 ulonglong4 {
    ulonglong at[4];
    struct { ulonglong x, y, z, w; };
};

union double1 {
    double at[1];
    double x;
};

union FK_VECTOR_16 double2 {
    double at[2];
    struct { double x, y; };
};

union FK_VECTOR_16 double3 {
    double at[3];
    struct { double x, y, z; };
};

union FK_VECTOR_16 double4 {
    double at[4];
    struct { double x, y, z, w; };
};

#undef FK_VECTOR
#undef FK_VECTOR_2
#undef FK_VECTOR_4
#undef FK_VECTOR_8
#undef FK_VECTOR_16

#endif /* __NVCC__ || __HIPCC__ */
#endif /* FK_VECTOR_TYPES */