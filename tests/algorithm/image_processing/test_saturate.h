/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Albert Andaluz
   Copyright 2025 Grup Mediapro S.L.U

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "fused_kernel/algorithms/image_processing/saturate.h"
#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/core/utils/type_to_string.h>
#include <fused_kernel/core/utils/vlimits.h>
#include <tests/operation_test_utils.h>

// uint[1234]->uint[1234]
void addUintUintTests() {

    // ADD_UNARY_TEST((0,200, 100), (0, 200, 100), fk::SaturateCast, uint, uint)

    ADD_UNARY_TEST((fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
                   (fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
                   fk::SaturateCast, uint1, uint1);

    ADD_UNARY_TEST((fk::minValue<uint2>, fk::maxValue<uint2> / static_cast<uint>(2), fk::maxValue<uint2>),
                   (fk::minValue<uint2>, fk::maxValue<uint2> / static_cast<uint>(2), fk::maxValue<uint2>),
                   fk::SaturateCast, uint2, uint2);

    ADD_UNARY_TEST((fk::minValue<uint3>, fk::maxValue<uint3> / static_cast<uint>(2), fk::maxValue<uint3>),
                   (fk::minValue<uint3>, fk::maxValue<uint3> / static_cast<uint>(2), fk::maxValue<uint3>),
                   fk::SaturateCast, uint3, uint3);

    ADD_UNARY_TEST((fk::minValue<uint4>, fk::maxValue<uint4> / static_cast<uint>(2), fk::maxValue<uint4>),
                   (fk::minValue<uint4>, fk::maxValue<uint4> / static_cast<uint>(2), fk::maxValue<uint4>),
                   fk::SaturateCast, uint4, uint4);
}

void addUintIntTests() {
    ADD_UNARY_TEST((fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
                   (fk::make_set<int1>(0), fk::maxValue<int1>, fk::maxValue<int1>), fk::SaturateCast, uint1, int1);

    ADD_UNARY_TEST((fk::minValue<uint2>, fk::maxValue<uint2> / static_cast<uint>(2), fk::maxValue<uint2>),
                   (fk::make_set<int2>(0), fk::maxValue<int2>, fk::maxValue<int2>), fk::SaturateCast, uint2, int2);

    ADD_UNARY_TEST((fk::minValue<uint3>, fk::maxValue<uint3> / static_cast<uint>(2), fk::maxValue<uint3>),
                   (fk::make_set<int3>(0), fk::maxValue<int3>, fk::maxValue<int3>), fk::SaturateCast, uint3, int3);

    ADD_UNARY_TEST((fk::minValue<uint4>, fk::maxValue<uint4> / static_cast<uint>(2), fk::maxValue<uint4>),
                   (fk::make_set<int4>(0), fk::maxValue<int4>, fk::maxValue<int4>), fk::SaturateCast, uint4, int4);
}

void addUintUCharTests() {
    ADD_UNARY_TEST((fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
                   (fk::make_set<uchar1>(0), fk::maxValue<uchar1>, fk::maxValue<uchar1>), fk::SaturateCast, uint1,
                   uchar1);

    ADD_UNARY_TEST((fk::minValue<uint2>, fk::maxValue<uint2> / static_cast<uint>(2), fk::maxValue<uint2>),
                   (fk::make_set<uchar2>(0), fk::maxValue<uchar2>, fk::maxValue<uchar2>), fk::SaturateCast, uint2,
                   uchar2);

    ADD_UNARY_TEST((fk::minValue<uint3>, fk::maxValue<uint3> / static_cast<uint>(2), fk::maxValue<uint3>),
                   (fk::make_set<uchar3>(0), fk::maxValue<uchar3>, fk::maxValue<uchar3>), fk::SaturateCast, uint3,
                   uchar3);

    ADD_UNARY_TEST((fk::minValue<uint4>, fk::maxValue<uint4> / static_cast<uint>(2), fk::maxValue<uint4>),
                   (fk::make_set<uchar4>(0), fk::maxValue<uchar4>, fk::maxValue<uchar4>), fk::SaturateCast, uint4,
                   uchar4);
}

void addUintCharTests() {
    ADD_UNARY_TEST((fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
                   (fk::make_set<char1>(0), fk::maxValue<char1>, fk::maxValue<char1>), fk::SaturateCast, uint1, char1);

    ADD_UNARY_TEST((fk::minValue<uint2>, fk::maxValue<uint2> / static_cast<uint>(2), fk::maxValue<uint2>),
                   (fk::make_set<char2>(0), fk::maxValue<char2>, fk::maxValue<char2>), fk::SaturateCast, uint2, char2);

    ADD_UNARY_TEST((fk::minValue<uint3>, fk::maxValue<uint3> / static_cast<uint>(2), fk::maxValue<uint3>),
                   (fk::make_set<char3>(0), fk::maxValue<char3>, fk::maxValue<char3>), fk::SaturateCast, uint3, char3);

    ADD_UNARY_TEST((fk::minValue<uint4>, fk::maxValue<uint4> / static_cast<uint>(2), fk::maxValue<uint4>),
                   (fk::make_set<char4>(0), fk::maxValue<char4>, fk::maxValue<char4>), fk::SaturateCast, uint4, char4);
}

void addUintUShortTests() {
    ADD_UNARY_TEST((fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
                   (fk::make_set<ushort1>(0), fk::maxValue<ushort1>, fk::maxValue<ushort1>), fk::SaturateCast, uint1,
                   ushort1);

    ADD_UNARY_TEST((fk::minValue<uint2>, fk::maxValue<uint2> / static_cast<uint>(2), fk::maxValue<uint2>),
                   (fk::make_set<ushort2>(0), fk::maxValue<ushort2>, fk::maxValue<ushort2>), fk::SaturateCast, uint2,
                   ushort2);

    ADD_UNARY_TEST((fk::minValue<uint3>, fk::maxValue<uint3> / static_cast<uint>(2), fk::maxValue<uint3>),
                   (fk::make_set<ushort3>(0), fk::maxValue<ushort3>, fk::maxValue<ushort3>), fk::SaturateCast, uint3,
                   ushort3);

    ADD_UNARY_TEST((fk::minValue<uint4>, fk::maxValue<uint4> / static_cast<uint>(2), fk::maxValue<uint4>),
                   (fk::make_set<ushort4>(0), fk::maxValue<ushort4>, fk::maxValue<ushort4>), fk::SaturateCast, uint4,
                   ushort4);
}

void addUintShortTests() {

    ADD_UNARY_TEST((fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
                   (fk::make_set<short1>(0), fk::maxValue<short1>, fk::maxValue<short1>), fk::SaturateCast, uint1,
                   short1);

    ADD_UNARY_TEST((fk::minValue<uint2>, fk::maxValue<uint2> / static_cast<uint>(2), fk::maxValue<uint2>),
                   (fk::make_set<short2>(0), fk::maxValue<short2>, fk::maxValue<short2>), fk::SaturateCast, uint2,
                   short2);

    ADD_UNARY_TEST((fk::minValue<uint3>, fk::maxValue<uint3> / static_cast<uint>(2), fk::maxValue<uint3>),
                   (fk::make_set<short3>(0), fk::maxValue<short3>, fk::maxValue<short3>), fk::SaturateCast, uint3,
                   short3);

    ADD_UNARY_TEST((fk::minValue<uint4>, fk::maxValue<uint4> / static_cast<uint>(2), fk::maxValue<uint4>),
                   (fk::make_set<short4>(0), fk::maxValue<short4>, fk::maxValue<short4>), fk::SaturateCast, uint4,
                   short4);
}

void addUintULongTests() {

    ADD_UNARY_TEST(
        (fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
        (fk::make_set<ulong1>(0), fk::maxValue<uint> / static_cast<uint>(2), fk::make_set<ulong1>(fk::maxValue<uint>)),
        fk::SaturateCast, uint1, ulong1);

    ADD_UNARY_TEST((fk::minValue<uint2>, fk::maxValue<uint2> / static_cast<uint>(2), fk::maxValue<uint2>),
                   (fk::make_set<ulong2>(0), fk::make_set<ulong2>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<ulong2>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint2, ulong2);

    ADD_UNARY_TEST((fk::minValue<uint3>, fk::maxValue<uint3> / static_cast<uint>(2), fk::maxValue<uint3>),
                   (fk::make_set<ulong3>(0), fk::make_set<ulong3>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<ulong3>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint3, ulong3);

    ADD_UNARY_TEST((fk::minValue<uint4>, fk::maxValue<uint4> / static_cast<uint>(2), fk::maxValue<uint4>),
                   (fk::make_set<ulong4>(0), fk::make_set<ulong4>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<ulong4>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint4, ulong4);
}

void addUintULongLongTests() {

    ADD_UNARY_TEST((fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
                   (fk::make_set<ulonglong1>(0), fk::maxValue<uint> / static_cast<uint>(2),
                    fk::make_set<ulonglong1>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint1, ulonglong1);

    ADD_UNARY_TEST((fk::minValue<uint2>, fk::maxValue<uint2> / static_cast<uint>(2), fk::maxValue<uint2>),
                   (fk::make_set<ulonglong2>(0), fk::make_set<ulonglong2>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<ulonglong2>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint2, ulonglong2);

    ADD_UNARY_TEST((fk::minValue<uint3>, fk::maxValue<uint3> / static_cast<uint>(2), fk::maxValue<uint3>),
                   (fk::make_set<ulonglong3>(0), fk::make_set<ulonglong3>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<ulonglong3>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint3, ulonglong3);

    ADD_UNARY_TEST((fk::minValue<uint4>, fk::maxValue<uint4> / static_cast<uint>(2), fk::maxValue<uint4>),
                   (fk::make_set<ulonglong4>(0), fk::make_set<ulonglong4>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<ulonglong4>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint4, ulonglong4);
}

void addUintLongTests() {

    ADD_UNARY_TEST((fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
                   (fk::make_set<long1>(0), fk::make_set<long1>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<long1>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint1, long1);

    ADD_UNARY_TEST((fk::minValue<uint2>, fk::maxValue<uint2> / static_cast<uint>(2), fk::maxValue<uint2>),
                   (fk::make_set<long2>(0), fk::make_set<long2>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<long2>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint2, long2);

    ADD_UNARY_TEST((fk::minValue<uint3>, fk::maxValue<uint3> / static_cast<uint>(2), fk::maxValue<uint3>),
                   (fk::make_set<long3>(0), fk::make_set<long3>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<long3>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint3, long3);

    ADD_UNARY_TEST((fk::minValue<uint4>, fk::maxValue<uint4> / static_cast<uint>(2), fk::maxValue<uint4>),
                   (fk::make_set<long4>(0), fk::make_set<long4>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<long4>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint4, long4);
}

void addUintLongLongTests() {

    ADD_UNARY_TEST((fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
                   (fk::make_set<longlong1>(0), fk::maxValue<uint> / static_cast<uint>(2),
                    fk::make_set<longlong1>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint1, longlong1);

    ADD_UNARY_TEST((fk::minValue<uint2>, fk::maxValue<uint2> / static_cast<uint>(2), fk::maxValue<uint2>),
                   (fk::make_set<longlong2>(0), fk::make_set<longlong2>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<longlong2>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint2, longlong2);

    ADD_UNARY_TEST((fk::minValue<uint3>, fk::maxValue<uint3> / static_cast<uint>(2), fk::maxValue<uint3>),
                   (fk::make_set<longlong3>(0), fk::make_set<longlong3>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<longlong3>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint3, longlong3);

    ADD_UNARY_TEST((fk::minValue<uint4>, fk::maxValue<uint4> / static_cast<uint>(2), fk::maxValue<uint4>),
                   (fk::make_set<longlong4>(0), fk::make_set<longlong4>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<longlong4>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint4, longlong4);
}

void addUintFloatTests() {
    ADD_UNARY_TEST((fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
                   (fk::make_set<float1>(fk::minValue<uint>), fk::maxValue<uint> / static_cast<float>(2),
                    fk::make_set<float1>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint1, float1);

    ADD_UNARY_TEST((fk::minValue<uint2>, fk::maxValue<uint2> / static_cast<uint>(2), fk::maxValue<uint2>),
                   (fk::make_set<float2>(fk::minValue<uint>),
                    fk::make_set<float2>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<float2>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint2, float2);

    ADD_UNARY_TEST((fk::minValue<uint3>, fk::maxValue<uint3> / static_cast<uint>(2), fk::maxValue<uint3>),
                   (fk::make_set<float3>(fk::minValue<uint>),
                    fk::make_set<float3>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<float3>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint3, float3);

    ADD_UNARY_TEST((fk::minValue<uint4>, fk::maxValue<uint4> / static_cast<uint>(2), fk::maxValue<uint4>),
                   (fk::make_set<float4>(fk::minValue<uint>),
                    fk::make_set<float4>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<float4>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint4, float4);
}

void addUintDoubleTests() {
    ADD_UNARY_TEST(
        (fk::minValue<uint1>, fk::make_set<uint1>({static_cast<uint>((fk::maxValue<uint> / 2))}), fk::maxValue<uint1>),
        (fk::make_set<double1>(fk::minValue<uint>), fk::make_set<double1>(fk::maxValue<uint> / static_cast<uint>(2)),
         fk::make_set<double1>(fk::maxValue<uint>)),
        fk::SaturateCast, uint1, double1);

    ADD_UNARY_TEST(
        (fk::minValue<uint1>, fk::make_set<uint1>({static_cast<uint>((fk::maxValue<uint> / 2))}), fk::maxValue<uint1>),
        (fk::make_set<double1>(fk::minValue<uint>), fk::make_set<double1>(fk::maxValue<uint> / static_cast<uint>(2)),
         fk::make_set<double1>(fk::maxValue<uint>)),
        fk::SaturateCast, uint1, double1);
    ADD_UNARY_TEST(
        (fk::minValue<uint1>, fk::make_set<uint1>({static_cast<uint>((fk::maxValue<uint> / 2))}), fk::maxValue<uint1>),
        (fk::make_set<double1>(fk::minValue<uint>), fk::make_set<double1>(fk::maxValue<uint> / static_cast<uint>(2)),
         fk::make_set<double1>(fk::maxValue<uint>)),
        fk::SaturateCast, uint1, double1);
    ADD_UNARY_TEST(
        (fk::minValue<uint1>, fk::make_set<uint1>({static_cast<uint>((fk::maxValue<uint> / 2))}), fk::maxValue<uint1>),
        (fk::make_set<double1>(fk::minValue<uint>), fk::make_set<double1>(fk::maxValue<uint> / static_cast<uint>(2)),
         fk::make_set<double1>(fk::maxValue<uint>)),
        fk::SaturateCast, uint1, double1);
}

// uint[1234]->uint[1234]
void addIntIntTests() {
    // ADD_UNARY_TEST((0,200, 100), (0, 200, 100), fk::SaturateCast, uint, uint)

    ADD_UNARY_TEST((fk::minValue<int1>, fk::maxValue<int1> / static_cast<int>(2), fk::maxValue<int1>),
                   (fk::minValue<int1>, fk::maxValue<int1> / static_cast<int>(2), fk::maxValue<int1>), fk::SaturateCast,
                   int1, int1);

    ADD_UNARY_TEST((fk::minValue<int2>, fk::maxValue<int2> / static_cast<int>(2), fk::maxValue<int2>),
                   (fk::minValue<int2>, fk::maxValue<int2> / static_cast<int>(2), fk::maxValue<int2>), fk::SaturateCast,
                   int2, int2);

    ADD_UNARY_TEST((fk::minValue<int3>, fk::maxValue<int3> / static_cast<int>(2), fk::maxValue<int3>),
                   (fk::minValue<int3>, fk::maxValue<int3> / static_cast<int>(2), fk::maxValue<int3>), fk::SaturateCast,
                   int3, int3);

    ADD_UNARY_TEST((fk::minValue<int4>, fk::maxValue<int4> / static_cast<int>(2), fk::maxValue<int4>),
                   (fk::minValue<int4>, fk::maxValue<int4> / static_cast<int>(2), fk::maxValue<int4>), fk::SaturateCast,
                   int4, int4);
}

void addIntUintTests() {

    ADD_UNARY_TEST((fk::minValue<int1>, fk::maxValue<int1> / static_cast<int>(2), fk::maxValue<int1>),
                   (fk::minValue<uint1>, fk::make_set<uint1>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<uint1>(fk::maxValue<int>)),
                   fk::SaturateCast, int1, uint1);
    ADD_UNARY_TEST((fk::minValue<int2>, fk::maxValue<int2> / static_cast<int>(2), fk::maxValue<int2>),
                   (fk::minValue<uint2>, fk::make_set<uint2>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<uint2>(fk::maxValue<int>)),
                   fk::SaturateCast, int2, uint2);

    ADD_UNARY_TEST((fk::minValue<int3>, fk::maxValue<int3> / static_cast<int>(2), fk::maxValue<int3>),
                   (fk::minValue<uint3>, fk::make_set<uint3>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<uint3>(fk::maxValue<int>)),
                   fk::SaturateCast, int3, uint3);

    ADD_UNARY_TEST((fk::minValue<int4>, fk::maxValue<int4> / static_cast<int>(2), fk::maxValue<int4>),
                   (fk::minValue<uint4>, fk::make_set<uint4>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<uint4>(fk::maxValue<int>)),
                   fk::SaturateCast, int4, uint4);
}

void addIntUcharTests() {

    ADD_UNARY_TEST((fk::minValue<int1>, fk::maxValue<int1> / static_cast<int>(2), fk::maxValue<int1>),
                   (fk::minValue<uchar1>, fk::make_set<uchar1>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<uchar1>(fk::maxValue<int>)),
                   fk::SaturateCast, int1, uchar1);

    ADD_UNARY_TEST(
        (fk::minValue<int2>, fk::maxValue<int2> / static_cast<int>(2), fk::maxValue<int2>),
        (fk::minValue<uchar2>, fk::make_set<uchar2>(fk::maxValue<int> / static_cast<int>(2)), fk::maxValue<uchar2>),
        fk::SaturateCast, int2, uchar2);

    ADD_UNARY_TEST(
        (fk::minValue<int3>, fk::maxValue<int3> / static_cast<int>(2), fk::maxValue<int3>),
        (fk::minValue<uchar3>, fk::make_set<uchar3>(fk::maxValue<int> / static_cast<int>(2)), fk::maxValue<uchar3>),
        fk::SaturateCast, int3, uchar3);

    ADD_UNARY_TEST(
        (fk::minValue<int4>, fk::maxValue<int4> / static_cast<int>(2), fk::maxValue<int4>),
        (fk::minValue<uchar4>, fk::make_set<uchar4>(fk::maxValue<int> / static_cast<int>(2)), fk::maxValue<uchar4>),
        fk::SaturateCast, int4, uchar4);
}

void addIntCharTests() {

    ADD_UNARY_TEST((fk::minValue<int1>, fk::maxValue<int1> / static_cast<int>(2), fk::maxValue<int1>),
                   (fk::minValue<char1>, fk::maxValue<char1>, fk::maxValue<char1>), fk::SaturateCast, int1, char1);

    ADD_UNARY_TEST((fk::minValue<int2>, fk::maxValue<int2> / static_cast<int>(2), fk::maxValue<int2>),
                   (fk::minValue<char2>, fk::maxValue<char2>, fk::maxValue<char2>), fk::SaturateCast, int2, char2);
    ADD_UNARY_TEST((fk::minValue<int3>, fk::maxValue<int3> / static_cast<int>(2), fk::maxValue<int3>),
                   (fk::minValue<char3>, fk::maxValue<char3>, fk::maxValue<char3>), fk::SaturateCast, int3, char3);
    ADD_UNARY_TEST((fk::minValue<int4>, fk::maxValue<int4> / static_cast<int>(2), fk::maxValue<int4>),
                   (fk::minValue<char4>, fk::maxValue<char4>, fk::maxValue<char4>), fk::SaturateCast, int4, char4);
}

void addIntShortTests() {

    ADD_UNARY_TEST((fk::minValue<int1>, fk::maxValue<int1> / static_cast<int>(2), fk::maxValue<int1>),
                   (fk::minValue<short1>, fk::maxValue<short1>, fk::maxValue<short1>), fk::SaturateCast, int1, short1);

    ADD_UNARY_TEST((fk::minValue<int2>, fk::maxValue<int2> / static_cast<int>(2), fk::maxValue<int2>),
                   (fk::minValue<short2>, fk::maxValue<short2>, fk::maxValue<short2>), fk::SaturateCast, int2, short2);
    ADD_UNARY_TEST((fk::minValue<int3>, fk::maxValue<int3> / static_cast<int>(2), fk::maxValue<int3>),
                   (fk::minValue<short3>, fk::maxValue<short3>, fk::maxValue<short3>), fk::SaturateCast, int3, short3);
    ADD_UNARY_TEST((fk::minValue<int4>, fk::maxValue<int4> / static_cast<int>(2), fk::maxValue<int4>),
                   (fk::minValue<short4>, fk::maxValue<short4>, fk::maxValue<short4>), fk::SaturateCast, int4, short4);
}

void addIntUShortTests() {

    ADD_UNARY_TEST((fk::minValue<int1>, fk::maxValue<int1> / static_cast<int>(2), fk::maxValue<int1>),
                   (fk::minValue<ushort1>, fk::maxValue<ushort1>, fk::maxValue<ushort1>), fk::SaturateCast, int1,
                   ushort1);

    ADD_UNARY_TEST((fk::minValue<int2>, fk::maxValue<int2> / static_cast<int>(2), fk::maxValue<int2>),
                   (fk::minValue<ushort2>, fk::maxValue<ushort2>, fk::maxValue<ushort2>), fk::SaturateCast, int2,
                   ushort2);
    ADD_UNARY_TEST((fk::minValue<int3>, fk::maxValue<int3> / static_cast<int>(2), fk::maxValue<int3>),
                   (fk::minValue<ushort3>, fk::maxValue<ushort3>, fk::maxValue<ushort3>), fk::SaturateCast, int3,
                   ushort3);
    ADD_UNARY_TEST((fk::minValue<int4>, fk::maxValue<int4> / static_cast<int>(2), fk::maxValue<int4>),
                   (fk::minValue<ushort4>, fk::maxValue<ushort4>, fk::maxValue<ushort4>), fk::SaturateCast, int4,
                   ushort4);
}

void addIntLongTests() {

    ADD_UNARY_TEST((fk::minValue<int1>, fk::maxValue<int1> / static_cast<int>(2), fk::maxValue<int1>),
                   (fk::make_set<long1>(fk::minValue<int>),
                    fk::make_set<long1>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<long1>(fk::maxValue<int>)),
                   fk::SaturateCast, int1, long1);

    ADD_UNARY_TEST((fk::minValue<int2>, fk::maxValue<int2> / static_cast<int>(2), fk::maxValue<int2>),
                   (fk::make_set<long2>(fk::minValue<int>),
                    fk::make_set<long2>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<long2>(fk::maxValue<int>)),
                   fk::SaturateCast, int2, long2);
    ADD_UNARY_TEST((fk::minValue<int3>, fk::maxValue<int3> / static_cast<int>(2), fk::maxValue<int3>),
                   (fk::make_set<long3>(fk::minValue<int>),
                    fk::make_set<long3>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<long3>(fk::maxValue<int>)),
                   fk::SaturateCast, int3, long3);

    ADD_UNARY_TEST((fk::minValue<int4>, fk::maxValue<int4> / static_cast<int>(2), fk::maxValue<int4>),
                   (fk::make_set<long4>(fk::minValue<int>),
                    fk::make_set<long4>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<long4>(fk::maxValue<int>)),
                   fk::SaturateCast, int4, long4);
}

void addIntULongTests() {

    ADD_UNARY_TEST((fk::minValue<int1>, fk::maxValue<int1> / static_cast<int>(2), fk::maxValue<int1>),
                   (fk::make_set<ulong1>(fk::minValue<int>),
                    fk::make_set<ulong1>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<ulong1>(fk::maxValue<int>)),
                   fk::SaturateCast, int1, ulong1);

    ADD_UNARY_TEST((fk::minValue<int2>, fk::maxValue<int2> / static_cast<int>(2), fk::maxValue<int2>),
                   (fk::make_set<ulong2>(fk::minValue<int>),
                    fk::make_set<ulong2>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<ulong2>(fk::maxValue<int>)),
                   fk::SaturateCast, int2, ulong2);

    ADD_UNARY_TEST((fk::minValue<int3>, fk::maxValue<int3> / static_cast<int>(2), fk::maxValue<int3>),
                   (fk::make_set<ulong3>(fk::minValue<int>),
                    fk::make_set<ulong3>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<ulong3>(fk::maxValue<int>)),
                   fk::SaturateCast, int3, ulong3);

    ADD_UNARY_TEST((fk::minValue<int4>, fk::maxValue<int4> / static_cast<int>(2), fk::maxValue<int4>),
                   (fk::make_set<ulong4>(fk::minValue<int>),
                    fk::make_set<ulong4>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<ulong4>(fk::maxValue<int>)),
                   fk::SaturateCast, int4, ulong4);
}

void addIntLongLongTests() {

    ADD_UNARY_TEST((fk::minValue<int1>, fk::maxValue<int1> / static_cast<int>(2), fk::maxValue<int1>),
                   (fk::make_set<long1>(fk::minValue<int>),
                    fk::make_set<long1>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<long1>(fk::maxValue<int>)),
                   fk::SaturateCast, int1, long1);

    ADD_UNARY_TEST((fk::minValue<int2>, fk::maxValue<int2> / static_cast<int>(2), fk::maxValue<int2>),
                   (fk::make_set<long2>(fk::minValue<int>),
                    fk::make_set<long2>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<long2>(fk::maxValue<int>)),
                   fk::SaturateCast, int2, long2);
    ADD_UNARY_TEST((fk::minValue<int3>, fk::maxValue<int3> / static_cast<int>(2), fk::maxValue<int3>),
                   (fk::make_set<long3>(fk::minValue<int>),
                    fk::make_set<long3>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<long3>(fk::maxValue<int>)),
                   fk::SaturateCast, int3, long3);

    ADD_UNARY_TEST((fk::minValue<int4>, fk::maxValue<int4> / static_cast<int>(2), fk::maxValue<int4>),
                   (fk::make_set<long4>(fk::minValue<int>),
                    fk::make_set<long4>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<long4>(fk::maxValue<int>)),
                   fk::SaturateCast, int4, long4);
}

void addIntULongLongTests() {

    ADD_UNARY_TEST((fk::minValue<int1>, fk::maxValue<int1> / static_cast<int>(2), fk::maxValue<int1>),
                   (fk::make_set<ulonglong1>(fk::minValue<int>),
                    fk::make_set<ulonglong1>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<ulonglong1>(fk::maxValue<int>)),
                   fk::SaturateCast, int1, ulonglong1);

    ADD_UNARY_TEST((fk::minValue<int2>, fk::maxValue<int2> / static_cast<int>(2), fk::maxValue<int2>),
                   (fk::make_set<ulonglong2>(fk::minValue<int>),
                    fk::make_set<ulonglong2>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<ulonglong2>(fk::maxValue<int>)),
                   fk::SaturateCast, int2, ulonglong2);

    ADD_UNARY_TEST((fk::minValue<int3>, fk::maxValue<int3> / static_cast<int>(2), fk::maxValue<int3>),
                   (fk::make_set<ulonglong3>(fk::minValue<int>),
                    fk::make_set<ulonglong3>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<ulonglong3>(fk::maxValue<int>)),
                   fk::SaturateCast, int3, ulonglong3);

    ADD_UNARY_TEST((fk::minValue<int4>, fk::maxValue<int4> / static_cast<int>(2), fk::maxValue<int4>),
                   (fk::make_set<ulonglong4>(fk::minValue<int>),
                    fk::make_set<ulonglong4>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<ulonglong4>(fk::maxValue<int>)),
                   fk::SaturateCast, int4, ulonglong4);
}

void addIntFloatTests() {
    ADD_UNARY_TEST((fk::minValue<int1>, fk::maxValue<int1> / static_cast<int>(2), fk::maxValue<int1>),
                   (fk::make_set<float1>(fk::minValue<int>), fk::maxValue<int> / static_cast<float>(2),
                    fk::make_set<float1>(fk::maxValue<int>)),
                   fk::SaturateCast, int1, float1);

    ADD_UNARY_TEST((fk::minValue<int2>, fk::maxValue<int2> / static_cast<int>(2), fk::maxValue<int2>),
                   (fk::make_set<float2>(fk::minValue<int>),
                    fk::make_set<float2>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<float2>(fk::maxValue<int>)),
                   fk::SaturateCast, int2, float2);

    ADD_UNARY_TEST((fk::minValue<int3>, fk::maxValue<int3> / static_cast<int>(2), fk::maxValue<int3>),
                   (fk::make_set<float3>(fk::minValue<int>),
                    fk::make_set<float3>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<float3>(fk::maxValue<int>)),
                   fk::SaturateCast, int3, float3);

    ADD_UNARY_TEST((fk::minValue<int4>, fk::maxValue<int4> / static_cast<int>(2), fk::maxValue<int4>),
                   (fk::make_set<float4>(fk::minValue<int>),
                    fk::make_set<float4>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<float4>(fk::maxValue<int>)),
                   fk::SaturateCast, int4, float4);
}

void addIntDoubleTests() {
    ADD_UNARY_TEST(
        (fk::minValue<int1>, fk::make_set<int1>({static_cast<int>((fk::maxValue<int> / 2))}), fk::maxValue<int1>),
        (fk::make_set<double1>(fk::minValue<int>), fk::make_set<double1>(fk::maxValue<int> / static_cast<int>(2)),
         fk::make_set<double1>(fk::maxValue<int>)),
        fk::SaturateCast, int1, double1);

    ADD_UNARY_TEST(
        (fk::minValue<int1>, fk::make_set<int1>({static_cast<int>((fk::maxValue<int> / 2))}), fk::maxValue<int1>),
        (fk::make_set<double1>(fk::minValue<int>), fk::make_set<double1>(fk::maxValue<int> / static_cast<int>(2)),
         fk::make_set<double1>(fk::maxValue<int>)),
        fk::SaturateCast, int1, double1);
    ADD_UNARY_TEST(
        (fk::minValue<int1>, fk::make_set<int1>({static_cast<int>((fk::maxValue<int> / 2))}), fk::maxValue<int1>),
        (fk::make_set<double1>(fk::minValue<int>), fk::make_set<double1>(fk::maxValue<int> / static_cast<int>(2)),
         fk::make_set<double1>(fk::maxValue<int>)),
        fk::SaturateCast, int1, double1);
    ADD_UNARY_TEST(
        (fk::minValue<int1>, fk::make_set<int1>({static_cast<int>((fk::maxValue<int> / 2))}), fk::maxValue<int1>),
        (fk::make_set<double1>(fk::minValue<int>), fk::make_set<double1>(fk::maxValue<int> / static_cast<int>(2)),
         fk::make_set<double1>(fk::maxValue<int>)),
        fk::SaturateCast, int1, double1);
}

void addCharIntTests() {
    ADD_UNARY_TEST(
        (fk::minValue<char1>, fk::make_set<char1>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char1>),
        (fk::make_set<int1>(fk::minValue<char>), fk::make_set<int1>(fk::maxValue<char>) / static_cast<int>(2),
         fk::make_set<int1>(fk::maxValue<char>)),
        fk::SaturateCast, char1, int1);

    ADD_UNARY_TEST(
        (fk::minValue<char2>, fk::make_set<char2>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char2>),
        (fk::make_set<int2>(fk::minValue<char>), fk::make_set<int2>(fk::maxValue<char>) / static_cast<int>(2),
         fk::make_set<int2>(fk::maxValue<char>)),
        fk::SaturateCast, char2, int2);

    ADD_UNARY_TEST(
        (fk::minValue<char3>, fk::make_set<char3>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char3>),
        (fk::make_set<int3>(fk::minValue<char>), fk::make_set<int3>(fk::maxValue<char>) / static_cast<int>(2),
         fk::make_set<int3>(fk::maxValue<char>)),
        fk::SaturateCast, char3, int3);

    ADD_UNARY_TEST(
        (fk::minValue<char4>, fk::make_set<char4>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char4>),
        (fk::make_set<int4>(fk::minValue<char>), fk::make_set<int4>(fk::maxValue<char>) / static_cast<int>(2),
         fk::make_set<int4>(fk::maxValue<char>)),
        fk::SaturateCast, char4, int4);
}

void addCharUIntTests() {

    ADD_UNARY_TEST((fk::minValue<char1>, fk::make_set<char1>({static_cast<char>((fk::maxValue<char1> / 2).x)}),
                    fk::maxValue<char1>),
                   (fk::make_set<uint1>(static_cast<uint>(0) / static_cast<int>(1)),
                    fk::make_set<uint1>(static_cast<uint>(fk::maxValue<char>) / static_cast<int>(2)),
                    fk::make_set<uint1>(static_cast<uint>(fk::maxValue<char>) / static_cast<int>(1))),
                   fk::SaturateCast, char1, uint1);

    ADD_UNARY_TEST((fk::minValue<char2>, fk::make_set<char2>({static_cast<char>((fk::maxValue<char2> / 2).x)}),
                    fk::maxValue<char2>),
                   (fk::make_set<uint2>(static_cast<uint>(0) / static_cast<int>(1)),
                    fk::make_set<uint2>(static_cast<uint>(fk::maxValue<char>) / static_cast<int>(2)),
                    fk::make_set<uint2>(static_cast<uint>(fk::maxValue<char>) / static_cast<int>(1))),
                   fk::SaturateCast, char2, uint2);

    ADD_UNARY_TEST((fk::minValue<char3>, fk::make_set<char3>({static_cast<char>((fk::maxValue<char3> / 2).x)}),
                    fk::maxValue<char3>),
                   (fk::make_set<uint3>(static_cast<uint>(0) / static_cast<int>(1)),
                    fk::make_set<uint3>(static_cast<uint>(fk::maxValue<char>) / static_cast<int>(2)),
                    fk::make_set<uint3>(static_cast<uint>(fk::maxValue<char>) / static_cast<int>(1))),
                   fk::SaturateCast, char3, uint3);

    ADD_UNARY_TEST((fk::minValue<char4>, fk::make_set<char4>({static_cast<char>((fk::maxValue<char4> / 2).x)}),
                    fk::maxValue<char4>),
                   (fk::make_set<uint4>(static_cast<uint>(0) / static_cast<int>(1)),
                    fk::make_set<uint4>(static_cast<uint>(fk::maxValue<char>) / static_cast<int>(2)),
                    fk::make_set<uint4>(static_cast<uint>(fk::maxValue<char>) / static_cast<int>(1))),
                   fk::SaturateCast, char4, uint4);
}

void addCharCharTests() {

    ADD_UNARY_TEST(
        (fk::minValue<char1>, fk::make_set<char1>({static_cast<char>(fk::maxValue<char> / 2)}), fk::maxValue<char1>),
        (fk::minValue<char1>, fk::make_set<char1>(fk::maxValue<char> / 2), fk::maxValue<char1>), fk::SaturateCast,
        char1, char1);

    ADD_UNARY_TEST(
        (fk::minValue<char2>, fk::make_set<char2>({static_cast<char>(fk::maxValue<char> / 2)}), fk::maxValue<char2>),
        (fk::minValue<char2>, fk::make_set<char2>(fk::maxValue<char> / 2), fk::maxValue<char2>), fk::SaturateCast,
        char2, char2);

    ADD_UNARY_TEST(
        (fk::minValue<char3>, fk::make_set<char3>({static_cast<char>(fk::maxValue<char> / 2)}), fk::maxValue<char3>),
        (fk::minValue<char3>, fk::make_set<char3>(fk::maxValue<char> / 2), fk::maxValue<char3>), fk::SaturateCast,
        char3, char3);

    ADD_UNARY_TEST(
        (fk::minValue<char4>, fk::make_set<char4>({static_cast<char>(fk::maxValue<char> / 2)}), fk::maxValue<char4>),
        (fk::minValue<char4>, fk::make_set<char4>(fk::maxValue<char> / 2), fk::maxValue<char4>), fk::SaturateCast,
        char4, char4);
}

void addCharUCharTests() {

    ADD_UNARY_TEST(
        (fk::minValue<char1>, fk::make_set<char1>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char1>),
        (fk::make_set<uchar1>(fk::minValue<uchar>), fk::make_set<uchar1>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<uchar1>(fk::maxValue<char>)),
        fk::SaturateCast, char1, uchar1);

    ADD_UNARY_TEST(
        (fk::minValue<char2>, fk::make_set<char2>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char2>),
        (fk::make_set<uchar2>(fk::minValue<uchar>), fk::make_set<uchar2>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<uchar2>(fk::maxValue<char>)),
        fk::SaturateCast, char2, uchar2);

    ADD_UNARY_TEST(
        (fk::minValue<char3>, fk::make_set<char3>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char3>),
        (fk::make_set<uchar3>(fk::minValue<uchar>), fk::make_set<uchar3>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<uchar3>(fk::maxValue<char>)),
        fk::SaturateCast, char3, uchar3);

    ADD_UNARY_TEST(
        (fk::minValue<char4>, fk::make_set<char4>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char4>),
        (fk::make_set<uchar4>(fk::minValue<uchar>), fk::make_set<uchar4>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<uchar4>(fk::maxValue<char>)),
        fk::SaturateCast, char4, uchar4);
}

void addCharShortTests() {

    ADD_UNARY_TEST(
        (fk::minValue<char1>, fk::make_set<char1>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char1>),
        (fk::make_set<short1>(fk::minValue<char>), fk::make_set<short1>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<short1>(fk::maxValue<char>)),
        fk::SaturateCast, char1, short1);

    ADD_UNARY_TEST(
        (fk::minValue<char2>, fk::make_set<char2>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char2>),
        (fk::make_set<short2>(fk::minValue<char>), fk::make_set<short2>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<short2>(fk::maxValue<char>)),
        fk::SaturateCast, char2, short2);

    ADD_UNARY_TEST(
        (fk::minValue<char3>, fk::make_set<char3>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char3>),
        (fk::make_set<short3>(fk::minValue<char>), fk::make_set<short3>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<short3>(fk::maxValue<char>)),
        fk::SaturateCast, char3, short3);

    ADD_UNARY_TEST(
        (fk::minValue<char4>, fk::make_set<char4>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char4>),
        (fk::make_set<short4>(fk::minValue<char>), fk::make_set<short4>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<short4>(fk::maxValue<char>)),
        fk::SaturateCast, char4, short4);
}

void addCharUShortTests() {

    ADD_UNARY_TEST(
        (fk::minValue<char1>, fk::make_set<char1>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char1>),
        (fk::make_set<ushort1>(fk::minValue<ushort>), fk::make_set<ushort1>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<ushort1>(fk::maxValue<char>)),
        fk::SaturateCast, char1, ushort1);

    ADD_UNARY_TEST(
        (fk::minValue<char2>, fk::make_set<char2>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char2>),
        (fk::make_set<ushort2>(fk::minValue<ushort>), fk::make_set<ushort2>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<ushort2>(fk::maxValue<char>)),
        fk::SaturateCast, char2, ushort2);

    ADD_UNARY_TEST(
        (fk::minValue<char3>, fk::make_set<char3>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char3>),
        (fk::make_set<ushort3>(fk::minValue<ushort>), fk::make_set<ushort3>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<ushort3>(fk::maxValue<char>)),
        fk::SaturateCast, char3, ushort3);

    ADD_UNARY_TEST(
        (fk::minValue<char4>, fk::make_set<char4>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char4>),
        (fk::make_set<ushort4>(fk::minValue<ushort>), fk::make_set<ushort4>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<ushort4>(fk::maxValue<char>)),
        fk::SaturateCast, char4, ushort4);
}

void addCharLongTests() {

    ADD_UNARY_TEST(
        (fk::minValue<char1>, fk::make_set<char1>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char1>),
        (fk::make_set<long1>(fk::minValue<char>), fk::make_set<long1>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<long1>(fk::maxValue<char>)),
        fk::SaturateCast, char1, long1);

    ADD_UNARY_TEST(
        (fk::minValue<char2>, fk::make_set<char2>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char2>),
        (fk::make_set<long2>(fk::minValue<char>), fk::make_set<long2>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<long2>(fk::maxValue<char>)),
        fk::SaturateCast, char2, long2);

    ADD_UNARY_TEST(
        (fk::minValue<char3>, fk::make_set<char3>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char3>),
        (fk::make_set<long3>(fk::minValue<char>), fk::make_set<long3>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<long3>(fk::maxValue<char>)),
        fk::SaturateCast, char3, long3);

    ADD_UNARY_TEST(
        (fk::minValue<char4>, fk::make_set<char4>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char4>),
        (fk::make_set<long4>(fk::minValue<char>), fk::make_set<long4>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<long4>(fk::maxValue<char>)),
        fk::SaturateCast, char4, long4);
}

void addCharULongTests() {

    ADD_UNARY_TEST(
        (fk::minValue<char1>, fk::make_set<char1>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char1>),
        (fk::make_set<ulong1>(fk::minValue<char>), fk::make_set<ulong1>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<ulong1>(fk::maxValue<char>)),
        fk::SaturateCast, char1, ulong1);

    ADD_UNARY_TEST(
        (fk::minValue<char2>, fk::make_set<char2>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char2>),
        (fk::make_set<ulong2>(fk::minValue<char>), fk::make_set<ulong2>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<ulong2>(fk::maxValue<char>)),
        fk::SaturateCast, char2, ulong2);

    ADD_UNARY_TEST(
        (fk::minValue<char3>, fk::make_set<char3>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char3>),
        (fk::make_set<ulong3>(fk::minValue<char>), fk::make_set<ulong3>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<ulong3>(fk::maxValue<char>)),
        fk::SaturateCast, char3, ulong3);

    ADD_UNARY_TEST(
        (fk::minValue<char4>, fk::make_set<char4>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char4>),
        (fk::make_set<ulong4>(fk::minValue<char>), fk::make_set<ulong4>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<ulong4>(fk::maxValue<char>)),
        fk::SaturateCast, char4, ulong4);
}

void addCharLongLongTests() {

    ADD_UNARY_TEST(
        (fk::minValue<char1>, fk::make_set<char1>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char1>),
        (fk::make_set<longlong1>(fk::minValue<char>), fk::make_set<longlong1>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<longlong1>(fk::maxValue<char>)),
        fk::SaturateCast, char1, longlong1);

    ADD_UNARY_TEST(
        (fk::minValue<char2>, fk::make_set<char2>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char2>),
        (fk::make_set<longlong2>(fk::minValue<char>), fk::make_set<longlong2>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<longlong2>(fk::maxValue<char>)),
        fk::SaturateCast, char2, longlong2);

    ADD_UNARY_TEST(
        (fk::minValue<char3>, fk::make_set<char3>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char3>),
        (fk::make_set<longlong3>(fk::minValue<char>), fk::make_set<longlong3>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<longlong3>(fk::maxValue<char>)),
        fk::SaturateCast, char3, longlong3);

    ADD_UNARY_TEST(
        (fk::minValue<char4>, fk::make_set<char4>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char4>),
        (fk::make_set<longlong4>(fk::minValue<char>), fk::make_set<longlong4>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<longlong4>(fk::maxValue<char>)),
        fk::SaturateCast, char4, longlong4);
}

void addCharULongLongTests() {

    ADD_UNARY_TEST(
        (fk::minValue<char1>, fk::make_set<char1>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char1>),
        (fk::make_set<ulonglong1>(fk::minValue<char>),
         fk::make_set<ulonglong1>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<ulonglong1>(fk::maxValue<char>)),
        fk::SaturateCast, char1, ulonglong1);

    ADD_UNARY_TEST(
        (fk::minValue<char2>, fk::make_set<char2>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char2>),
        (fk::make_set<ulonglong2>(fk::minValue<char>),
         fk::make_set<ulonglong2>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<ulonglong2>(fk::maxValue<char>)),
        fk::SaturateCast, char2, ulonglong2);

    ADD_UNARY_TEST(
        (fk::minValue<char3>, fk::make_set<char3>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char3>),
        (fk::make_set<ulonglong3>(fk::minValue<char>),
         fk::make_set<ulonglong3>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<ulonglong3>(fk::maxValue<char>)),
        fk::SaturateCast, char3, ulonglong3);

    ADD_UNARY_TEST(
        (fk::minValue<char4>, fk::make_set<char4>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char4>),
        (fk::make_set<ulonglong4>(fk::minValue<char>),
         fk::make_set<ulonglong4>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<ulonglong4>(fk::maxValue<char>)),
        fk::SaturateCast, char4, ulonglong4);
}

void addCharFloatTests() {

    ADD_UNARY_TEST(
        (fk::minValue<char1>, fk::make_set<char1>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char1>),
        (fk::make_set<float1>(fk::minValue<char>), fk::make_set<float1>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<float1>(fk::maxValue<char>)),
        fk::SaturateCast, char1, float1);

    ADD_UNARY_TEST(
        (fk::minValue<char2>, fk::make_set<char2>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char2>),
        (fk::make_set<float2>(fk::minValue<char>), fk::make_set<float2>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<float2>(fk::maxValue<char>)),
        fk::SaturateCast, char2, float2);

    ADD_UNARY_TEST(
        (fk::minValue<char3>, fk::make_set<char3>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char3>),
        (fk::make_set<float3>(fk::minValue<char>), fk::make_set<float3>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<float3>(fk::maxValue<char>)),
        fk::SaturateCast, char3, float3);

    ADD_UNARY_TEST(
        (fk::minValue<char4>, fk::make_set<char4>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char4>),
        (fk::make_set<float4>(fk::minValue<char>), fk::make_set<float4>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<float4>(fk::maxValue<char>)),
        fk::SaturateCast, char4, float4);
}

void addCharDoubleTests() {

    ADD_UNARY_TEST(
        (fk::minValue<char1>, fk::make_set<char1>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char1>),
        (fk::make_set<double1>(fk::minValue<char>), fk::make_set<double1>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<double1>(fk::maxValue<char>)),
        fk::SaturateCast, char1, double1);

    ADD_UNARY_TEST(
        (fk::minValue<char2>, fk::make_set<char2>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char2>),
        (fk::make_set<double2>(fk::minValue<char>), fk::make_set<double2>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<double2>(fk::maxValue<char>)),
        fk::SaturateCast, char2, double2);

    ADD_UNARY_TEST(
        (fk::minValue<char3>, fk::make_set<char3>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char3>),
        (fk::make_set<double3>(fk::minValue<char>), fk::make_set<double3>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<double3>(fk::maxValue<char>)),
        fk::SaturateCast, char3, double3);

    ADD_UNARY_TEST(
        (fk::minValue<char4>, fk::make_set<char4>({static_cast<char>((fk::maxValue<char> / 2))}), fk::maxValue<char4>),
        (fk::make_set<double4>(fk::minValue<char>), fk::make_set<double4>(fk::maxValue<char> / static_cast<int>(2)),
         fk::make_set<double4>(fk::maxValue<char>)),
        fk::SaturateCast, char4, double4);
}

void addUCharIntTests() {
    ADD_UNARY_TEST((fk::minValue<uchar1>, fk::make_set<uchar1>({static_cast<int>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar1>),
                   (fk::make_set<int1>(fk::minValue<uchar>),
                    fk::make_set<int1>(fk::maxValue<uchar>) / static_cast<int>(2),
                    fk::make_set<int1>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar1, int1);

    ADD_UNARY_TEST((fk::minValue<uchar2>, fk::make_set<uchar2>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar2>),
                   (fk::make_set<int2>(fk::minValue<uchar>),
                    fk::make_set<int2>(fk::maxValue<uchar>) / static_cast<int>(2),
                    fk::make_set<int2>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar2, int2);

    ADD_UNARY_TEST((fk::minValue<uchar3>, fk::make_set<uchar3>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar3>),
                   (fk::make_set<int3>(fk::minValue<uchar>),
                    fk::make_set<int3>(fk::maxValue<uchar>) / static_cast<int>(2),
                    fk::make_set<int3>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar3, int3);

    ADD_UNARY_TEST((fk::minValue<uchar4>, fk::make_set<uchar4>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar4>),
                   (fk::make_set<int4>(fk::minValue<uchar>),
                    fk::make_set<int4>(fk::maxValue<uchar>) / static_cast<int>(2),
                    fk::make_set<int4>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar4, int4);
}

void addUCharUIntTests() {

    ADD_UNARY_TEST((fk::minValue<uchar1>, fk::make_set<uchar1>({static_cast<uchar>((fk::maxValue<uchar1> / 2).x)}),
                    fk::maxValue<uchar1>),
                   (fk::make_set<uint1>(static_cast<uint>(0) / static_cast<int>(1)),
                    fk::make_set<uint1>(static_cast<uint>(fk::maxValue<uchar>) / static_cast<int>(2)),
                    fk::make_set<uint1>(static_cast<uint>(fk::maxValue<uchar>) / static_cast<int>(1))),
                   fk::SaturateCast, uchar1, uint1);

    ADD_UNARY_TEST((fk::minValue<uchar2>, fk::make_set<uchar2>({static_cast<uchar>((fk::maxValue<uchar2> / 2).x)}),
                    fk::maxValue<uchar2>),
                   (fk::make_set<uint2>(static_cast<uint>(0) / static_cast<int>(1)),
                    fk::make_set<uint2>(static_cast<uint>(fk::maxValue<uchar>) / static_cast<int>(2)),
                    fk::make_set<uint2>(static_cast<uint>(fk::maxValue<uchar>) / static_cast<int>(1))),
                   fk::SaturateCast, uchar2, uint2);

    ADD_UNARY_TEST((fk::minValue<uchar3>, fk::make_set<uchar3>({static_cast<uchar>((fk::maxValue<uchar3> / 2).x)}),
                    fk::maxValue<uchar3>),
                   (fk::make_set<uint3>(static_cast<uint>(0) / static_cast<int>(1)),
                    fk::make_set<uint3>(static_cast<uint>(fk::maxValue<uchar>) / static_cast<int>(2)),
                    fk::make_set<uint3>(static_cast<uint>(fk::maxValue<uchar>) / static_cast<int>(1))),
                   fk::SaturateCast, uchar3, uint3);

    ADD_UNARY_TEST((fk::minValue<uchar4>, fk::make_set<uchar4>({static_cast<uchar>((fk::maxValue<uchar4> / 2).x)}),
                    fk::maxValue<uchar4>),
                   (fk::make_set<uint4>(static_cast<uint>(0) / static_cast<int>(1)),
                    fk::make_set<uint4>(static_cast<uint>(fk::maxValue<uchar>) / static_cast<int>(2)),
                    fk::make_set<uint4>(static_cast<uint>(fk::maxValue<uchar>) / static_cast<int>(1))),
                   fk::SaturateCast, uchar4, uint4);
}

void addUcharUcharTests() {

    ADD_UNARY_TEST(
        (fk::minValue<uchar1>, fk::make_set<uchar1>(fk::maxValue<uchar> / static_cast<int>(2)), fk::maxValue<uchar1>),
        (fk::minValue<uchar1>, fk::make_set<uchar1>(fk::maxValue<uchar> / static_cast<int>(2)), fk::maxValue<uchar1>),
        fk::SaturateCast, uchar1, uchar1);

    ADD_UNARY_TEST(
        (fk::minValue<uchar2>, fk::make_set<uchar2>(fk::maxValue<uchar> / static_cast<int>(2)), fk::maxValue<uchar2>),
        (fk::minValue<uchar2>, fk::make_set<uchar2>(fk::maxValue<uchar> / static_cast<int>(2)), fk::maxValue<uchar2>),
        fk::SaturateCast, uchar2, uchar2);
    ADD_UNARY_TEST(
        (fk::minValue<uchar3>, fk::make_set<uchar3>(fk::maxValue<uchar> / static_cast<int>(2)), fk::maxValue<uchar3>),
        (fk::minValue<uchar3>, fk::make_set<uchar3>(fk::maxValue<uchar> / static_cast<int>(2)), fk::maxValue<uchar3>),
        fk::SaturateCast, uchar3, uchar3);
    ADD_UNARY_TEST(
        (fk::minValue<uchar4>, fk::make_set<uchar4>(fk::maxValue<uchar> / static_cast<int>(2)), fk::maxValue<uchar4>),
        (fk::minValue<uchar4>, fk::make_set<uchar4>(fk::maxValue<uchar> / static_cast<int>(2)), fk::maxValue<uchar4>),
        fk::SaturateCast, uchar4, uchar4);
}

void addUCharShortTests() {

    ADD_UNARY_TEST((fk::minValue<uchar1>, fk::make_set<uchar1>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar1>),
                   (fk::make_set<short1>(fk::minValue<uchar>),
                    fk::make_set<short1>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<short1>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar1, short1);

    ADD_UNARY_TEST((fk::minValue<uchar2>, fk::make_set<uchar2>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar2>),
                   (fk::make_set<short2>(fk::minValue<uchar>),
                    fk::make_set<short2>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<short2>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar2, short2);

    ADD_UNARY_TEST((fk::minValue<uchar3>, fk::make_set<uchar3>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar3>),
                   (fk::make_set<short3>(fk::minValue<uchar>),
                    fk::make_set<short3>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<short3>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar3, short3);

    ADD_UNARY_TEST((fk::minValue<uchar4>, fk::make_set<uchar4>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar4>),
                   (fk::make_set<short4>(fk::minValue<uchar>),
                    fk::make_set<short4>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<short4>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar4, short4);
}

void addUCharUShortTests() {

    ADD_UNARY_TEST((fk::minValue<uchar1>, fk::make_set<uchar1>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar1>),
                   (fk::make_set<ushort1>(fk::minValue<ushort>),
                    fk::make_set<ushort1>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<ushort1>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar1, ushort1);

    ADD_UNARY_TEST((fk::minValue<uchar2>, fk::make_set<uchar2>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar2>),
                   (fk::make_set<ushort2>(fk::minValue<ushort>),
                    fk::make_set<ushort2>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<ushort2>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar2, ushort2);

    ADD_UNARY_TEST((fk::minValue<uchar3>, fk::make_set<uchar3>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar3>),
                   (fk::make_set<ushort3>(fk::minValue<ushort>),
                    fk::make_set<ushort3>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<ushort3>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar3, ushort3);

    ADD_UNARY_TEST((fk::minValue<uchar4>, fk::make_set<uchar4>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar4>),
                   (fk::make_set<ushort4>(fk::minValue<ushort>),
                    fk::make_set<ushort4>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<ushort4>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar4, ushort4);
}

void addUCharLongTests() {

    ADD_UNARY_TEST((fk::minValue<uchar1>, fk::make_set<uchar1>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar1>),
                   (fk::make_set<long1>(fk::minValue<uchar>),
                    fk::make_set<long1>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<long1>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar1, long1);

    ADD_UNARY_TEST((fk::minValue<uchar2>, fk::make_set<uchar2>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar2>),
                   (fk::make_set<long2>(fk::minValue<uchar>),
                    fk::make_set<long2>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<long2>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar2, long2);

    ADD_UNARY_TEST((fk::minValue<uchar3>, fk::make_set<uchar3>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar3>),
                   (fk::make_set<long3>(fk::minValue<uchar>),
                    fk::make_set<long3>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<long3>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar3, long3);

    ADD_UNARY_TEST((fk::minValue<uchar4>, fk::make_set<uchar4>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar4>),
                   (fk::make_set<long4>(fk::minValue<uchar>),
                    fk::make_set<long4>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<long4>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar4, long4);
}

void addUCharULongTests() {

    ADD_UNARY_TEST((fk::minValue<uchar1>, fk::make_set<uchar1>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar1>),
                   (fk::make_set<ulong1>(fk::minValue<ulong>),
                    fk::make_set<ulong1>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<ulong1>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar1, ulong1);

    ADD_UNARY_TEST((fk::minValue<uchar2>, fk::make_set<uchar2>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar2>),
                   (fk::make_set<ulong2>(fk::minValue<ulong>),
                    fk::make_set<ulong2>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<ulong2>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar2, ulong2);

    ADD_UNARY_TEST((fk::minValue<uchar3>, fk::make_set<uchar3>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar3>),
                   (fk::make_set<ulong3>(fk::minValue<ulong>),
                    fk::make_set<ulong3>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<ulong3>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar3, ulong3);

    ADD_UNARY_TEST((fk::minValue<uchar4>, fk::make_set<uchar4>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar4>),
                   (fk::make_set<ulong4>(fk::minValue<ulong>),
                    fk::make_set<ulong4>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<ulong4>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar4, ulong4);
}

void addUCharLongLongTests() {

    ADD_UNARY_TEST((fk::minValue<uchar1>, fk::make_set<uchar1>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar1>),
                   (fk::make_set<longlong1>(fk::minValue<uchar>),
                    fk::make_set<longlong1>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<longlong1>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar1, longlong1);

    ADD_UNARY_TEST((fk::minValue<uchar2>, fk::make_set<uchar2>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar2>),
                   (fk::make_set<longlong2>(fk::minValue<uchar>),
                    fk::make_set<longlong2>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<longlong2>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar2, longlong2);

    ADD_UNARY_TEST((fk::minValue<uchar3>, fk::make_set<uchar3>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar3>),
                   (fk::make_set<longlong3>(fk::minValue<uchar>),
                    fk::make_set<longlong3>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<longlong3>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar3, longlong3);

    ADD_UNARY_TEST((fk::minValue<uchar4>, fk::make_set<uchar4>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar4>),
                   (fk::make_set<longlong4>(fk::minValue<uchar>),
                    fk::make_set<longlong4>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<longlong4>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar4, longlong4);
}

void addUCharULongLongTests() {

    ADD_UNARY_TEST((fk::minValue<uchar1>, fk::make_set<uchar1>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar1>),
                   (fk::make_set<ulonglong1>(fk::minValue<ulonglong>),
                    fk::make_set<ulonglong1>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<ulonglong1>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar1, ulonglong1);

    ADD_UNARY_TEST((fk::minValue<uchar2>, fk::make_set<uchar2>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar2>),
                   (fk::make_set<ulonglong2>(fk::minValue<ulonglong>),
                    fk::make_set<ulonglong2>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<ulonglong2>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar2, ulonglong2);

    ADD_UNARY_TEST((fk::minValue<uchar3>, fk::make_set<uchar3>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar3>),
                   (fk::make_set<ulonglong3>(fk::minValue<ulonglong>),
                    fk::make_set<ulonglong3>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<ulonglong3>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar3, ulonglong3);

    ADD_UNARY_TEST((fk::minValue<uchar4>, fk::make_set<uchar4>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar4>),
                   (fk::make_set<ulonglong4>(fk::minValue<ulonglong>),
                    fk::make_set<ulonglong4>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<ulonglong4>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar4, ulonglong4);
}

void addUCharFloatTests() {

    ADD_UNARY_TEST((fk::minValue<uchar1>, fk::make_set<uchar1>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar1>),
                   (fk::make_set<float1>(fk::minValue<uchar>),
                    fk::make_set<float1>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<float1>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar1, float1);

    ADD_UNARY_TEST((fk::minValue<uchar2>, fk::make_set<uchar2>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar2>),
                   (fk::make_set<float2>(fk::minValue<uchar>),
                    fk::make_set<float2>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<float2>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar2, float2);

    ADD_UNARY_TEST((fk::minValue<uchar3>, fk::make_set<uchar3>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar3>),
                   (fk::make_set<float3>(fk::minValue<uchar>),
                    fk::make_set<float3>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<float3>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar3, float3);

    ADD_UNARY_TEST((fk::minValue<uchar4>, fk::make_set<uchar4>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar4>),
                   (fk::make_set<float4>(fk::minValue<uchar>),
                    fk::make_set<float4>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<float4>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar4, float4);
}

void addUCharDoubleTests() {

    ADD_UNARY_TEST((fk::minValue<uchar1>, fk::make_set<uchar1>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar1>),
                   (fk::make_set<double1>(fk::minValue<uchar>),
                    fk::make_set<double1>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<double1>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar1, double1);

    ADD_UNARY_TEST((fk::minValue<uchar2>, fk::make_set<uchar2>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar2>),
                   (fk::make_set<double2>(fk::minValue<uchar>),
                    fk::make_set<double2>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<double2>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar2, double2);

    ADD_UNARY_TEST((fk::minValue<uchar3>, fk::make_set<uchar3>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar3>),
                   (fk::make_set<double3>(fk::minValue<uchar>),
                    fk::make_set<double3>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<double3>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar3, double3);

    ADD_UNARY_TEST((fk::minValue<uchar4>, fk::make_set<uchar4>({static_cast<uchar>((fk::maxValue<uchar> / 2))}),
                    fk::maxValue<uchar4>),
                   (fk::make_set<double4>(fk::minValue<uchar>),
                    fk::make_set<double4>(fk::maxValue<uchar> / static_cast<int>(2)),
                    fk::make_set<double4>(fk::maxValue<uchar>)),
                   fk::SaturateCast, uchar4, double4);
}

void addShortIntTests() {
    ADD_UNARY_TEST((fk::minValue<short1>, fk::make_set<short1>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short1>),
                   (fk::make_set<int1>(fk::minValue<short>),
                    fk::make_set<int1>(fk::maxValue<short>) / static_cast<int>(2),
                    fk::make_set<int1>(fk::maxValue<short>)),
                   fk::SaturateCast, short1, int1);

    ADD_UNARY_TEST((fk::minValue<short2>, fk::make_set<short2>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short2>),
                   (fk::make_set<int2>(fk::minValue<short>),
                    fk::make_set<int2>(fk::maxValue<short>) / static_cast<int>(2),
                    fk::make_set<int2>(fk::maxValue<short>)),
                   fk::SaturateCast, short2, int2);

    ADD_UNARY_TEST((fk::minValue<short3>, fk::make_set<short3>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short3>),
                   (fk::make_set<int3>(fk::minValue<short>),
                    fk::make_set<int3>(fk::maxValue<short>) / static_cast<int>(2),
                    fk::make_set<int3>(fk::maxValue<short>)),
                   fk::SaturateCast, short3, int3);

    ADD_UNARY_TEST((fk::minValue<short4>, fk::make_set<short4>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short4>),
                   (fk::make_set<int4>(fk::minValue<short>),
                    fk::make_set<int4>(fk::maxValue<short>) / static_cast<int>(2),
                    fk::make_set<int4>(fk::maxValue<short>)),
                   fk::SaturateCast, short4, int4);
}

void addShortUIntTests() {

    ADD_UNARY_TEST((fk::minValue<short1>, fk::make_set<short1>({static_cast<short>((fk::maxValue<short1> / 2).x)}),
                    fk::maxValue<short1>),
                   (fk::make_set<uint1>(static_cast<uint>(0) / static_cast<int>(1)),
                    fk::make_set<uint1>(static_cast<uint>(fk::maxValue<short>) / static_cast<int>(2)),
                    fk::make_set<uint1>(static_cast<uint>(fk::maxValue<short>) / static_cast<int>(1))),
                   fk::SaturateCast, short1, uint1);

    ADD_UNARY_TEST((fk::minValue<short2>, fk::make_set<short2>({static_cast<short>((fk::maxValue<short2> / 2).x)}),
                    fk::maxValue<short2>),
                   (fk::make_set<uint2>(static_cast<uint>(0) / static_cast<int>(1)),
                    fk::make_set<uint2>(static_cast<uint>(fk::maxValue<short>) / static_cast<int>(2)),
                    fk::make_set<uint2>(static_cast<uint>(fk::maxValue<short>) / static_cast<int>(1))),
                   fk::SaturateCast, short2, uint2);

    ADD_UNARY_TEST((fk::minValue<short3>, fk::make_set<short3>({static_cast<short>((fk::maxValue<short3> / 2).x)}),
                    fk::maxValue<short3>),
                   (fk::make_set<uint3>(static_cast<uint>(0) / static_cast<int>(1)),
                    fk::make_set<uint3>(static_cast<uint>(fk::maxValue<short>) / static_cast<int>(2)),
                    fk::make_set<uint3>(static_cast<uint>(fk::maxValue<short>) / static_cast<int>(1))),
                   fk::SaturateCast, short3, uint3);

    ADD_UNARY_TEST((fk::minValue<short4>, fk::make_set<short4>({static_cast<short>((fk::maxValue<short4> / 2).x)}),
                    fk::maxValue<short4>),
                   (fk::make_set<uint4>(static_cast<uint>(0) / static_cast<int>(1)),
                    fk::make_set<uint4>(static_cast<uint>(fk::maxValue<short>) / static_cast<int>(2)),
                    fk::make_set<uint4>(static_cast<uint>(fk::maxValue<short>) / static_cast<int>(1))),
                   fk::SaturateCast, short4, uint4);
}

void addShortCharTests() {

    ADD_UNARY_TEST(
        (fk::minValue<short1>, fk::make_set<short1>(fk::maxValue<short> / static_cast<short>(2)), fk::maxValue<short1>),
        (fk::minValue<char1>, fk::maxValue<char1>, fk::maxValue<char1>), fk::SaturateCast, short1, char1);

    ADD_UNARY_TEST(
        (fk::minValue<short2>, fk::make_set<short2>(fk::maxValue<short> / static_cast<short>(2)), fk::maxValue<short2>),
        (fk::minValue<char2>, fk::maxValue<char2>, fk::maxValue<char2>), fk::SaturateCast, short2, char2);

    ADD_UNARY_TEST(
        (fk::minValue<short3>, fk::make_set<short3>(fk::maxValue<short> / static_cast<short>(2)), fk::maxValue<short3>),
        (fk::minValue<char3>, fk::maxValue<char3>, fk::maxValue<char3>), fk::SaturateCast, short3, char3);

    ADD_UNARY_TEST(
        (fk::minValue<short4>, fk::make_set<short4>(fk::maxValue<short> / static_cast<short>(2)), fk::maxValue<short4>),
        (fk::minValue<char4>, fk::maxValue<char4>, fk::maxValue<char4>), fk::SaturateCast, short4, char4);
}

void addShortUCharTests() {

    ADD_UNARY_TEST(
        (fk::minValue<short1>, fk::make_set<short1>(fk::maxValue<short> / static_cast<short>(2)), fk::maxValue<short1>),
        (fk::minValue<uchar1>, fk::maxValue<uchar1>, fk::maxValue<uchar1>), fk::SaturateCast, short1, uchar1);

    ADD_UNARY_TEST(
        (fk::minValue<short2>, fk::make_set<short2>(fk::maxValue<short> / static_cast<short>(2)), fk::maxValue<short2>),
        (fk::minValue<uchar2>, fk::maxValue<uchar2>, fk::maxValue<uchar2>), fk::SaturateCast, short2, uchar2);

    ADD_UNARY_TEST(
        (fk::minValue<short3>, fk::make_set<short3>(fk::maxValue<short> / static_cast<short>(2)), fk::maxValue<short3>),
        (fk::minValue<uchar3>, fk::maxValue<uchar3>, fk::maxValue<uchar3>), fk::SaturateCast, short3, uchar3);

    ADD_UNARY_TEST(
        (fk::minValue<short4>, fk::make_set<short4>(fk::maxValue<short> / static_cast<short>(2)), fk::maxValue<short4>),
        (fk::minValue<uchar4>, fk::maxValue<uchar4>, fk::maxValue<uchar4>), fk::SaturateCast, short4, uchar4);
}

void addShortShortTests() {

    ADD_UNARY_TEST((fk::minValue<short1>, fk::make_set<short1>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short1>),
                   (fk::make_set<short1>(fk::minValue<short>),
                    fk::make_set<short1>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<short1>(fk::maxValue<short>)),
                   fk::SaturateCast, short1, short1);

    ADD_UNARY_TEST((fk::minValue<short2>, fk::make_set<short2>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short2>),
                   (fk::make_set<short2>(fk::minValue<short>),
                    fk::make_set<short2>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<short2>(fk::maxValue<short>)),
                   fk::SaturateCast, short2, short2);

    ADD_UNARY_TEST((fk::minValue<short3>, fk::make_set<short3>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short3>),
                   (fk::make_set<short3>(fk::minValue<short>),
                    fk::make_set<short3>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<short3>(fk::maxValue<short>)),
                   fk::SaturateCast, short3, short3);

    ADD_UNARY_TEST((fk::minValue<short4>, fk::make_set<short4>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short4>),
                   (fk::make_set<short4>(fk::minValue<short>),
                    fk::make_set<short4>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<short4>(fk::maxValue<short>)),
                   fk::SaturateCast, short4, short4);
}

void addShortUShortTests() {

    ADD_UNARY_TEST((fk::minValue<short1>, fk::make_set<short1>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short1>),
                   (fk::make_set<ushort1>(fk::minValue<ushort>),
                    fk::make_set<ushort1>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<ushort1>(fk::maxValue<short>)),
                   fk::SaturateCast, short1, ushort1);

    ADD_UNARY_TEST((fk::minValue<short2>, fk::make_set<short2>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short2>),
                   (fk::make_set<ushort2>(fk::minValue<ushort>),
                    fk::make_set<ushort2>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<ushort2>(fk::maxValue<short>)),
                   fk::SaturateCast, short2, ushort2);

    ADD_UNARY_TEST((fk::minValue<short3>, fk::make_set<short3>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short3>),
                   (fk::make_set<ushort3>(fk::minValue<ushort>),
                    fk::make_set<ushort3>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<ushort3>(fk::maxValue<short>)),
                   fk::SaturateCast, short3, ushort3);

    ADD_UNARY_TEST((fk::minValue<short4>, fk::make_set<short4>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short4>),
                   (fk::make_set<ushort4>(fk::minValue<ushort>),
                    fk::make_set<ushort4>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<ushort4>(fk::maxValue<short>)),
                   fk::SaturateCast, short4, ushort4);
}

void addShortLongTests() {

    ADD_UNARY_TEST((fk::minValue<short1>, fk::make_set<short1>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short1>),
                   (fk::make_set<long1>(fk::minValue<short>),
                    fk::make_set<long1>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<long1>(fk::maxValue<short>)),
                   fk::SaturateCast, short1, long1);

    ADD_UNARY_TEST((fk::minValue<short2>, fk::make_set<short2>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short2>),
                   (fk::make_set<long2>(fk::minValue<short>),
                    fk::make_set<long2>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<long2>(fk::maxValue<short>)),
                   fk::SaturateCast, short2, long2);

    ADD_UNARY_TEST((fk::minValue<short3>, fk::make_set<short3>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short3>),
                   (fk::make_set<long3>(fk::minValue<short>),
                    fk::make_set<long3>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<long3>(fk::maxValue<short>)),
                   fk::SaturateCast, short3, long3);

    ADD_UNARY_TEST((fk::minValue<short4>, fk::make_set<short4>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short4>),
                   (fk::make_set<long4>(fk::minValue<short>),
                    fk::make_set<long4>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<long4>(fk::maxValue<short>)),
                   fk::SaturateCast, short4, long4);
}

void addShortULongTests() {
    ADD_UNARY_TEST((fk::minValue<short1>, fk::make_set<short1>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short1>),
                   (fk::make_set<ulong1>(fk::minValue<short>),
                    fk::make_set<ulong1>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<ulong1>(fk::maxValue<short>)),
                   fk::SaturateCast, short1, ulong1);

    ADD_UNARY_TEST((fk::minValue<short2>, fk::make_set<short2>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short2>),
                   (fk::make_set<ulong2>(fk::minValue<short>),
                    fk::make_set<ulong2>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<ulong2>(fk::maxValue<short>)),
                   fk::SaturateCast, short2, ulong2);

    ADD_UNARY_TEST((fk::minValue<short3>, fk::make_set<short3>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short3>),
                   (fk::make_set<ulong3>(fk::minValue<short>),
                    fk::make_set<ulong3>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<ulong3>(fk::maxValue<short>)),
                   fk::SaturateCast, short3, ulong3);

    ADD_UNARY_TEST((fk::minValue<short4>, fk::make_set<short4>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short4>),
                   (fk::make_set<ulong4>(fk::minValue<short>),
                    fk::make_set<ulong4>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<ulong4>(fk::maxValue<short>)),
                   fk::SaturateCast, short4, ulong4);
}

void addShortLongLongTests() {

    ADD_UNARY_TEST((fk::minValue<short1>, fk::make_set<short1>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short1>),
                   (fk::make_set<longlong1>(fk::minValue<short>),
                    fk::make_set<longlong1>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<longlong1>(fk::maxValue<short>)),
                   fk::SaturateCast, short1, longlong1);

    ADD_UNARY_TEST((fk::minValue<short2>, fk::make_set<short2>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short2>),
                   (fk::make_set<longlong2>(fk::minValue<short>),
                    fk::make_set<longlong2>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<longlong2>(fk::maxValue<short>)),
                   fk::SaturateCast, short2, longlong2);

    ADD_UNARY_TEST((fk::minValue<short3>, fk::make_set<short3>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short3>),
                   (fk::make_set<longlong3>(fk::minValue<short>),
                    fk::make_set<longlong3>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<longlong3>(fk::maxValue<short>)),
                   fk::SaturateCast, short3, longlong3);

    ADD_UNARY_TEST((fk::minValue<short4>, fk::make_set<short4>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short4>),
                   (fk::make_set<longlong4>(fk::minValue<short>),
                    fk::make_set<longlong4>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<longlong4>(fk::maxValue<short>)),
                   fk::SaturateCast, short4, longlong4);
}

void addShortULongLongTests() {

    ADD_UNARY_TEST((fk::minValue<short1>, fk::make_set<short1>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short1>),
                   (fk::make_set<ulonglong1>(fk::minValue<short>),
                    fk::make_set<ulonglong1>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<ulonglong1>(fk::maxValue<short>)),
                   fk::SaturateCast, short1, ulonglong1);

    ADD_UNARY_TEST((fk::minValue<short2>, fk::make_set<short2>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short2>),
                   (fk::make_set<ulonglong2>(fk::minValue<short>),
                    fk::make_set<ulonglong2>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<ulonglong2>(fk::maxValue<short>)),
                   fk::SaturateCast, short2, ulonglong2);

    ADD_UNARY_TEST((fk::minValue<short3>, fk::make_set<short3>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short3>),
                   (fk::make_set<ulonglong3>(fk::minValue<short>),
                    fk::make_set<ulonglong3>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<ulonglong3>(fk::maxValue<short>)),
                   fk::SaturateCast, short3, ulonglong3);

    ADD_UNARY_TEST((fk::minValue<short4>, fk::make_set<short4>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short4>),
                   (fk::make_set<ulonglong4>(fk::minValue<short>),
                    fk::make_set<ulonglong4>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<ulonglong4>(fk::maxValue<short>)),
                   fk::SaturateCast, short4, ulonglong4);
}

void addShortFloatTests() {

    ADD_UNARY_TEST((fk::minValue<short1>, fk::make_set<short1>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short1>),
                   (fk::make_set<float1>(fk::minValue<short>),
                    fk::make_set<float1>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<float1>(fk::maxValue<short>)),
                   fk::SaturateCast, short1, float1);

    ADD_UNARY_TEST((fk::minValue<short2>, fk::make_set<short2>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short2>),
                   (fk::make_set<float2>(fk::minValue<short>),
                    fk::make_set<float2>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<float2>(fk::maxValue<short>)),
                   fk::SaturateCast, short2, float2);

    ADD_UNARY_TEST((fk::minValue<short3>, fk::make_set<short3>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short3>),
                   (fk::make_set<float3>(fk::minValue<short>),
                    fk::make_set<float3>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<float3>(fk::maxValue<short>)),
                   fk::SaturateCast, short3, float3);

    ADD_UNARY_TEST((fk::minValue<short4>, fk::make_set<short4>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short4>),
                   (fk::make_set<float4>(fk::minValue<short>),
                    fk::make_set<float4>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<float4>(fk::maxValue<short>)),
                   fk::SaturateCast, short4, float4);
}

void addShortDoubleTests() {

    ADD_UNARY_TEST((fk::minValue<short1>, fk::make_set<short1>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short1>),
                   (fk::make_set<double1>(fk::minValue<short>),
                    fk::make_set<double1>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<double1>(fk::maxValue<short>)),
                   fk::SaturateCast, short1, double1);

    ADD_UNARY_TEST((fk::minValue<short2>, fk::make_set<short2>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short2>),
                   (fk::make_set<double2>(fk::minValue<short>),
                    fk::make_set<double2>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<double2>(fk::maxValue<short>)),
                   fk::SaturateCast, short2, double2);

    ADD_UNARY_TEST((fk::minValue<short3>, fk::make_set<short3>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short3>),
                   (fk::make_set<double3>(fk::minValue<short>),
                    fk::make_set<double3>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<double3>(fk::maxValue<short>)),
                   fk::SaturateCast, short3, double3);

    ADD_UNARY_TEST((fk::minValue<short4>, fk::make_set<short4>({static_cast<short>((fk::maxValue<short> / 2))}),
                    fk::maxValue<short4>),
                   (fk::make_set<double4>(fk::minValue<short>),
                    fk::make_set<double4>(fk::maxValue<short> / static_cast<int>(2)),
                    fk::make_set<double4>(fk::maxValue<short>)),
                   fk::SaturateCast, short4, double4);
}

void addUShortIntTests() {
    ADD_UNARY_TEST((fk::minValue<ushort1>, fk::make_set<ushort1>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort1>),
                   (fk::make_set<int1>(fk::minValue<ushort>),
                    fk::make_set<int1>(fk::maxValue<ushort>) / static_cast<int>(2),
                    fk::make_set<int1>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort1, int1);

    ADD_UNARY_TEST((fk::minValue<ushort2>, fk::make_set<ushort2>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort2>),
                   (fk::make_set<int2>(fk::minValue<ushort>),
                    fk::make_set<int2>(fk::maxValue<ushort>) / static_cast<int>(2),
                    fk::make_set<int2>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort2, int2);

    ADD_UNARY_TEST((fk::minValue<ushort3>, fk::make_set<ushort3>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort3>),
                   (fk::make_set<int3>(fk::minValue<ushort>),
                    fk::make_set<int3>(fk::maxValue<ushort>) / static_cast<int>(2),
                    fk::make_set<int3>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort3, int3);

    ADD_UNARY_TEST((fk::minValue<ushort4>, fk::make_set<ushort4>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort4>),
                   (fk::make_set<int4>(fk::minValue<ushort>),
                    fk::make_set<int4>(fk::maxValue<ushort>) / static_cast<int>(2),
                    fk::make_set<int4>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort4, int4);
}

void addUShortUIntTests() {

    ADD_UNARY_TEST((fk::minValue<ushort1>, fk::make_set<ushort1>({static_cast<ushort>((fk::maxValue<ushort1> / 2).x)}),
                    fk::maxValue<ushort1>),
                   (fk::make_set<uint1>(static_cast<uint>(0) / static_cast<int>(1)),
                    fk::make_set<uint1>(static_cast<uint>(fk::maxValue<ushort>) / static_cast<int>(2)),
                    fk::make_set<uint1>(static_cast<uint>(fk::maxValue<ushort>) / static_cast<int>(1))),
                   fk::SaturateCast, ushort1, uint1);

    ADD_UNARY_TEST((fk::minValue<ushort2>, fk::make_set<ushort2>({static_cast<ushort>((fk::maxValue<ushort2> / 2).x)}),
                    fk::maxValue<ushort2>),
                   (fk::make_set<uint2>(static_cast<uint>(0) / static_cast<int>(1)),
                    fk::make_set<uint2>(static_cast<uint>(fk::maxValue<ushort>) / static_cast<int>(2)),
                    fk::make_set<uint2>(static_cast<uint>(fk::maxValue<ushort>) / static_cast<int>(1))),
                   fk::SaturateCast, ushort2, uint2);

    ADD_UNARY_TEST((fk::minValue<ushort3>, fk::make_set<ushort3>({static_cast<ushort>((fk::maxValue<ushort3> / 2).x)}),
                    fk::maxValue<ushort3>),
                   (fk::make_set<uint3>(static_cast<uint>(0) / static_cast<int>(1)),
                    fk::make_set<uint3>(static_cast<uint>(fk::maxValue<ushort>) / static_cast<int>(2)),
                    fk::make_set<uint3>(static_cast<uint>(fk::maxValue<ushort>) / static_cast<int>(1))),
                   fk::SaturateCast, ushort3, uint3);

    ADD_UNARY_TEST((fk::minValue<ushort4>, fk::make_set<ushort4>({static_cast<ushort>((fk::maxValue<ushort4> / 2).x)}),
                    fk::maxValue<ushort4>),
                   (fk::make_set<uint4>(static_cast<uint>(0) / static_cast<int>(1)),
                    fk::make_set<uint4>(static_cast<uint>(fk::maxValue<ushort>) / static_cast<int>(2)),
                    fk::make_set<uint4>(static_cast<uint>(fk::maxValue<ushort>) / static_cast<int>(1))),
                   fk::SaturateCast, ushort4, uint4);
}

void addUShortCharTests() {

    ADD_UNARY_TEST(
        (fk::minValue<ushort1>, fk::make_set<ushort1>(fk::maxValue<ushort> / static_cast<ushort>(2)),
         fk::maxValue<ushort1>),
        (fk::make_set<char1>(fk::minValue<ushort>), fk::make_set<char1>(fk::maxValue<char>), fk::maxValue<char>),
        fk::SaturateCast, ushort1, char1);

    ADD_UNARY_TEST(
        (fk::minValue<ushort2>, fk::make_set<ushort2>(fk::maxValue<ushort> / static_cast<ushort>(2)),
         fk::maxValue<ushort2>),
        (fk::make_set<char2>(fk::minValue<ushort>), fk::make_set<char2>(fk::maxValue<char>), fk::maxValue<char>),
        fk::SaturateCast, ushort2, char2);

    ADD_UNARY_TEST(
        (fk::minValue<ushort3>, fk::make_set<ushort3>(fk::maxValue<ushort> / static_cast<ushort>(2)),
         fk::maxValue<ushort3>),
        (fk::make_set<char3>(fk::minValue<ushort>), fk::make_set<char3>(fk::maxValue<char>), fk::maxValue<char>),
        fk::SaturateCast, ushort3, char3);

    ADD_UNARY_TEST(
        (fk::minValue<ushort4>, fk::make_set<ushort4>(fk::maxValue<ushort> / static_cast<ushort>(2)),
         fk::maxValue<ushort4>),
        (fk::make_set<char4>(fk::minValue<ushort>), fk::make_set<char4>(fk::maxValue<char>), fk::maxValue<char>),
        fk::SaturateCast, ushort4, char4);
}

void addUShortUCharTests() {

    ADD_UNARY_TEST((fk::minValue<ushort1>, fk::make_set<ushort1>(fk::maxValue<ushort> / static_cast<ushort>(2)),
                    fk::maxValue<ushort1>),
                   (fk::minValue<uchar1>, fk::maxValue<uchar1>, fk::maxValue<uchar1>), fk::SaturateCast, ushort1,
                   uchar1);

    ADD_UNARY_TEST((fk::minValue<ushort2>, fk::make_set<ushort2>(fk::maxValue<ushort> / static_cast<ushort>(2)),
                    fk::maxValue<ushort2>),
                   (fk::minValue<uchar2>, fk::maxValue<uchar2>, fk::maxValue<uchar2>), fk::SaturateCast, ushort2,
                   uchar2);

    ADD_UNARY_TEST((fk::minValue<ushort3>, fk::make_set<ushort3>(fk::maxValue<ushort> / static_cast<ushort>(2)),
                    fk::maxValue<ushort3>),
                   (fk::minValue<uchar3>, fk::maxValue<uchar3>, fk::maxValue<uchar3>), fk::SaturateCast, ushort3,
                   uchar3);

    ADD_UNARY_TEST((fk::minValue<ushort4>, fk::make_set<ushort4>(fk::maxValue<ushort> / static_cast<ushort>(2)),
                    fk::maxValue<ushort4>),
                   (fk::minValue<uchar4>, fk::maxValue<uchar4>, fk::maxValue<uchar4>), fk::SaturateCast, ushort4,
                   uchar4);
}

void addUShortShortTests() {

    ADD_UNARY_TEST((fk::minValue<ushort1>, fk::make_set<ushort1>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort1>),
                   (fk::make_set<short1>(fk::minValue<ushort>),
                    fk::make_set<short1>(fk::maxValue<ushort> / static_cast<int>(2)), fk::maxValue<short1>),
                   fk::SaturateCast, ushort1, short1);

    ADD_UNARY_TEST((fk::minValue<ushort2>, fk::make_set<ushort2>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort2>),
                   (fk::make_set<short2>(fk::minValue<ushort>),
                    fk::make_set<short2>(fk::maxValue<ushort> / static_cast<int>(2)), fk::maxValue<short2>),
                   fk::SaturateCast, ushort2, short2);

    ADD_UNARY_TEST((fk::minValue<ushort3>, fk::make_set<ushort3>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort3>),
                   (fk::make_set<short3>(fk::minValue<ushort>),
                    fk::make_set<short3>(fk::maxValue<ushort> / static_cast<int>(2)), fk::maxValue<short3>),
                   fk::SaturateCast, ushort3, short3);

    ADD_UNARY_TEST((fk::minValue<ushort4>, fk::make_set<ushort4>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort4>),
                   (fk::make_set<short4>(fk::minValue<ushort>),
                    fk::make_set<short4>(fk::maxValue<ushort> / static_cast<int>(2)), fk::maxValue<short4>),
                   fk::SaturateCast, ushort4, short4);
}

void addUShortUShortTests() {

    ADD_UNARY_TEST((fk::minValue<ushort1>, fk::make_set<ushort1>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort1>),
                   (fk::make_set<ushort1>(fk::minValue<ushort>),
                    fk::make_set<ushort1>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<ushort1>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort1, ushort1);

    ADD_UNARY_TEST((fk::minValue<ushort2>, fk::make_set<ushort2>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort2>),
                   (fk::make_set<ushort2>(fk::minValue<ushort>),
                    fk::make_set<ushort2>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<ushort2>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort2, ushort2);

    ADD_UNARY_TEST((fk::minValue<ushort3>, fk::make_set<ushort3>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort3>),
                   (fk::make_set<ushort3>(fk::minValue<ushort>),
                    fk::make_set<ushort3>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<ushort3>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort3, ushort3);

    ADD_UNARY_TEST((fk::minValue<ushort4>, fk::make_set<ushort4>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort4>),
                   (fk::make_set<ushort4>(fk::minValue<ushort>),
                    fk::make_set<ushort4>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<ushort4>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort4, ushort4);
}

void addUShortLongTests() {

    ADD_UNARY_TEST((fk::minValue<ushort1>, fk::make_set<ushort1>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort1>),
                   (fk::make_set<long1>(fk::minValue<ushort>),
                    fk::make_set<long1>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<long1>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort1, long1);

    ADD_UNARY_TEST((fk::minValue<ushort2>, fk::make_set<ushort2>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort2>),
                   (fk::make_set<long2>(fk::minValue<ushort>),
                    fk::make_set<long2>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<long2>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort2, long2);

    ADD_UNARY_TEST((fk::minValue<ushort3>, fk::make_set<ushort3>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort3>),
                   (fk::make_set<long3>(fk::minValue<ushort>),
                    fk::make_set<long3>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<long3>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort3, long3);

    ADD_UNARY_TEST((fk::minValue<ushort4>, fk::make_set<ushort4>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort4>),
                   (fk::make_set<long4>(fk::minValue<ushort>),
                    fk::make_set<long4>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<long4>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort4, long4);
}

void addUShortULongTests() {
    ADD_UNARY_TEST((fk::minValue<ushort1>, fk::make_set<ushort1>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort1>),
                   (fk::make_set<ulong1>(fk::minValue<ushort>),
                    fk::make_set<ulong1>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<ulong1>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort1, ulong1);

    ADD_UNARY_TEST((fk::minValue<ushort2>, fk::make_set<ushort2>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort2>),
                   (fk::make_set<ulong2>(fk::minValue<ushort>),
                    fk::make_set<ulong2>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<ulong2>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort2, ulong2);

    ADD_UNARY_TEST((fk::minValue<ushort3>, fk::make_set<ushort3>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort3>),
                   (fk::make_set<ulong3>(fk::minValue<ushort>),
                    fk::make_set<ulong3>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<ulong3>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort3, ulong3);

    ADD_UNARY_TEST((fk::minValue<ushort4>, fk::make_set<ushort4>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort4>),
                   (fk::make_set<ulong4>(fk::minValue<ushort>),
                    fk::make_set<ulong4>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<ulong4>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort4, ulong4);
}

void addUShortLongLongTests() {

    ADD_UNARY_TEST((fk::minValue<ushort1>, fk::make_set<ushort1>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort1>),
                   (fk::make_set<longlong1>(fk::minValue<ushort>),
                    fk::make_set<longlong1>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<longlong1>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort1, longlong1);

    ADD_UNARY_TEST((fk::minValue<ushort2>, fk::make_set<ushort2>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort2>),
                   (fk::make_set<longlong2>(fk::minValue<ushort>),
                    fk::make_set<longlong2>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<longlong2>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort2, longlong2);

    ADD_UNARY_TEST((fk::minValue<ushort3>, fk::make_set<ushort3>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort3>),
                   (fk::make_set<longlong3>(fk::minValue<ushort>),
                    fk::make_set<longlong3>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<longlong3>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort3, longlong3);

    ADD_UNARY_TEST((fk::minValue<ushort4>, fk::make_set<ushort4>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort4>),
                   (fk::make_set<longlong4>(fk::minValue<ushort>),
                    fk::make_set<longlong4>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<longlong4>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort4, longlong4);
}

void addUShortULongLongTests() {

    ADD_UNARY_TEST((fk::minValue<ushort1>, fk::make_set<ushort1>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort1>),
                   (fk::make_set<ulonglong1>(fk::minValue<ushort>),
                    fk::make_set<ulonglong1>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<ulonglong1>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort1, ulonglong1);

    ADD_UNARY_TEST((fk::minValue<ushort2>, fk::make_set<ushort2>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort2>),
                   (fk::make_set<ulonglong2>(fk::minValue<ushort>),
                    fk::make_set<ulonglong2>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<ulonglong2>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort2, ulonglong2);

    ADD_UNARY_TEST((fk::minValue<ushort3>, fk::make_set<ushort3>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort3>),
                   (fk::make_set<ulonglong3>(fk::minValue<ushort>),
                    fk::make_set<ulonglong3>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<ulonglong3>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort3, ulonglong3);

    ADD_UNARY_TEST((fk::minValue<ushort4>, fk::make_set<ushort4>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort4>),
                   (fk::make_set<ulonglong4>(fk::minValue<ushort>),
                    fk::make_set<ulonglong4>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<ulonglong4>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort4, ulonglong4);
}

void addUShortFloatTests() {

    ADD_UNARY_TEST((fk::minValue<ushort1>, fk::make_set<ushort1>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort1>),
                   (fk::make_set<float1>(fk::minValue<ushort>),
                    fk::make_set<float1>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<float1>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort1, float1);

    ADD_UNARY_TEST((fk::minValue<ushort2>, fk::make_set<ushort2>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort2>),
                   (fk::make_set<float2>(fk::minValue<ushort>),
                    fk::make_set<float2>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<float2>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort2, float2);

    ADD_UNARY_TEST((fk::minValue<ushort3>, fk::make_set<ushort3>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort3>),
                   (fk::make_set<float3>(fk::minValue<ushort>),
                    fk::make_set<float3>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<float3>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort3, float3);

    ADD_UNARY_TEST((fk::minValue<ushort4>, fk::make_set<ushort4>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort4>),
                   (fk::make_set<float4>(fk::minValue<ushort>),
                    fk::make_set<float4>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<float4>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort4, float4);
}

void addUShortDoubleTests() {

    ADD_UNARY_TEST((fk::minValue<ushort1>, fk::make_set<ushort1>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort1>),
                   (fk::make_set<double1>(fk::minValue<ushort>),
                    fk::make_set<double1>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<double1>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort1, double1);

    ADD_UNARY_TEST((fk::minValue<ushort2>, fk::make_set<ushort2>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort2>),
                   (fk::make_set<double2>(fk::minValue<ushort>),
                    fk::make_set<double2>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<double2>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort2, double2);

    ADD_UNARY_TEST((fk::minValue<ushort3>, fk::make_set<ushort3>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort3>),
                   (fk::make_set<double3>(fk::minValue<ushort>),
                    fk::make_set<double3>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<double3>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort3, double3);

    ADD_UNARY_TEST((fk::minValue<ushort4>, fk::make_set<ushort4>({static_cast<ushort>((fk::maxValue<ushort> / 2))}),
                    fk::maxValue<ushort4>),
                   (fk::make_set<double4>(fk::minValue<ushort>),
                    fk::make_set<double4>(fk::maxValue<ushort> / static_cast<int>(2)),
                    fk::make_set<double4>(fk::maxValue<ushort>)),
                   fk::SaturateCast, ushort4, double4);
}

START_ADDING_TESTS

addIntIntTests();
addIntUintTests();
addIntCharTests();
addIntUcharTests();
addIntShortTests();
addIntUShortTests();
addIntLongTests();
addIntULongTests();
addIntLongLongTests();
addIntULongLongTests();
addIntFloatTests();
addIntDoubleTests();

// uint

addUintIntTests();
addUintUintTests();
addUintCharTests();
addUintUCharTests();
addUintShortTests();
addUintUShortTests();
addUintLongTests();
addUintLongLongTests();
addUintULongTests();
addUintULongLongTests();
addUintFloatTests();
addUintDoubleTests();

// Char
addCharIntTests();
addCharUIntTests();
addCharCharTests();
addCharUCharTests();
addCharShortTests();
addCharUShortTests();
addCharLongTests();
addCharULongTests();
addCharLongLongTests();
addCharULongLongTests();
addCharFloatTests();
addCharDoubleTests();

// uchar
addUCharIntTests();
addUCharUIntTests();
addUcharUcharTests();
addUCharShortTests();
addUCharUShortTests();
addUCharLongTests();
addUCharULongTests();
addUCharLongLongTests();
addUCharULongLongTests();
addUCharFloatTests();
addUCharDoubleTests();

// short
addShortIntTests();
addShortUIntTests();
addShortCharTests();
addShortUCharTests();
addShortShortTests();
addShortUShortTests();
addShortLongTests();
addShortULongTests();
addShortLongLongTests();
addShortULongLongTests();
addShortFloatTests();
addShortDoubleTests();

// ushort
addUShortIntTests();
addUShortUIntTests();
addUShortCharTests();
addUShortUCharTests();
addUShortShortTests();
addUShortUShortTests();
addUShortLongTests();
addUShortULongTests();
addUShortLongLongTests();
addUShortULongLongTests();
addUShortFloatTests();
addUShortDoubleTests();

STOP_ADDING_TESTS

// You can add more tests for other type combinations as needed.
int launch() {

    RUN_ALL_TESTS
};