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
    bool x;
    bool at[1];
};

union bool2 {
    bool x, y;
    bool at[2];
};

union bool3 {
    bool x, y, z;
    bool at[3];
};

union bool4 {
    bool x, y, z, w;
    bool at[4];
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
    signed char x;
    signed char at[1];
};

union uchar1 {
    uchar x;
    uchar at[1];
};

union FK_VECTOR_2 char2 {
    signed char x, y;
    signed char at[2];
};

union FK_VECTOR_2 uchar2 {
    uchar x, y;
    uchar at[2];
};

union char3 {
    signed char x, y, z;
    signed char at[3];
};

union uchar3 {
    uchar x, y, z;
    uchar at[3];
};

union FK_VECTOR_4 char4 {
    signed char x, y, z, w;
    signed char at[4];
};

union FK_VECTOR_4 uchar4 {
    uchar x, y, z, w;
    uchar at[4];
};

union short1 {
    short x;
    short at[1];
};

union ushort1 {
    ushort x;
    ushort at[1];
};

union FK_VECTOR_4 short2{
    short x, y;
    short at[2];
};

union FK_VECTOR_4 ushort2{
    ushort x, y;
    ushort at[2];
};

union short3 {
    short x, y, z;
    short at[3];
};

union ushort3 {
    ushort x, y, z;
    ushort at[3];
};

union FK_VECTOR_8 short4 {
    short x, y, z, w;
    short at[4];
};

union FK_VECTOR_8 ushort4 {
    ushort x, y, z, w;
    ushort at[4];
};

union int1 {
    int x;
    int at[1];
};

union uint1 {
    unsigned int x;
    unsigned int at[1];
};

union FK_VECTOR_8 int2 {
    int x, y;
    int at[2];
};

union FK_VECTOR_8 uint2 {
    unsigned int x, y;
    unsigned int at[2];
};

union int3 {
    int x, y, z;
    int at[3];
};

union uint3 {
    unsigned int x, y, z;
    unsigned int at[3];
};

union FK_VECTOR_16 int4 {
    int x, y, z, w;
    int at[4];
};

union FK_VECTOR_16 uint4 {
    unsigned int x, y, z, w;
    unsigned int at[4];
};

union long1 {
    long int x;
    long int at[1];
};

union ulong1 {
    ulong x;
    ulong at[1];
};

union FK_VECTOR_(2*sizeof(long int))  long2 {
    long int x, y;
    long int at[2];
};

union FK_VECTOR_(2*sizeof(unsigned long int)) ulong2 {
    ulong x, y;
    ulong at[2];
};

union long3 {
    long int x, y, z;
    long int at[3];
};

union ulong3 {
    ulong x, y, z;
    ulong at[3];
};

union FK_VECTOR_16 long4 {
    long int x, y, z, w;
    long int at[4];
};

union FK_VECTOR_16 ulong4 {
    ulong x, y, z, w;
    ulong at[4];
};

union float1 {
    float x;
    float at[1];
};

union FK_VECTOR_8 float2 {
    float x, y;
    float at[2];
};

union float3 {
    float x, y, z;
    float at[3];
};

union FK_VECTOR_16 float4 {
    float x, y, z, w;
    float at[4];
};

union longlong1 {
    long long int x;
    long long int at[1];
};

union ulonglong1 {
    ulonglong x;
    ulonglong at[1];
};

union FK_VECTOR_16 longlong2 {
    long long int x, y;
    long long int at[2];
};

union FK_VECTOR_16 ulonglong2 {
    ulonglong x, y;
    ulonglong at[2];
};

union longlong3 {
    long long int x, y, z;
    long long int at[3];
};

union ulonglong3 {
    ulonglong x, y, z;
    ulonglong at[3];
};

union FK_VECTOR_16 longlong4 {
    long long int x, y, z, w;
    long long int at[4];
};

union FK_VECTOR_16 ulonglong4 {
    ulonglong x, y, z, w;
    ulonglong at[4];
};

union double1 {
    double x;
    double at[1];
};

union FK_VECTOR_16 double2 {
    double x, y;
    double at[2];
};

union FK_VECTOR_16 double3 {
    double x, y, z;
    double at[3];
};

union FK_VECTOR_16 double4 {
    double x, y, z, w;
    double at[4];
};

#undef FK_VECTOR
#undef FK_VECTOR_2
#undef FK_VECTOR_4
#undef FK_VECTOR_8
#undef FK_VECTOR_16

#endif /* __NVCC__ || __HIPCC__ */
#endif /* FK_VECTOR_TYPES */