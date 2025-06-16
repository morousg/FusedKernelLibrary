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
    ADD_UNARY_TEST((fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
                   (fk::make_set<double1>(fk::minValue<uint>), fk::maxValue<uint> / static_cast<double>(2),
                    fk::make_set<double1>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint1, double1);

    ADD_UNARY_TEST((fk::minValue<uint2>, fk::maxValue<uint2> / static_cast<uint>(2), fk::maxValue<uint2>),
                   (fk::make_set<double2>(fk::minValue<uint>),
                    fk::make_set<double2>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<double2>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint2, double2);

    ADD_UNARY_TEST((fk::minValue<uint3>, fk::maxValue<uint3> / static_cast<uint>(2), fk::maxValue<uint3>),
                   (fk::make_set<double3>(fk::minValue<uint>),
                    fk::make_set<double3>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<double3>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint3, double3);

    ADD_UNARY_TEST((fk::minValue<uint4>, fk::maxValue<uint4> / static_cast<uint>(2), fk::maxValue<uint4>),
                   (fk::make_set<double4>(fk::minValue<uint>),
                    fk::make_set<double4>(fk::maxValue<uint> / static_cast<uint>(2)),
                    fk::make_set<double4>(fk::maxValue<uint>)),
                   fk::SaturateCast, uint4, double4);
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
    ADD_UNARY_TEST((fk::minValue<int1>, fk::maxValue<int1> / static_cast<int>(2), fk::maxValue<int1>),
                   (fk::make_set<double1>(fk::minValue<int>), fk::maxValue<int> / static_cast<double>(2),
                    fk::make_set<double1>(fk::maxValue<int>)),
                   fk::SaturateCast, int1, double1);

    ADD_UNARY_TEST((fk::minValue<int2>, fk::maxValue<int2> / static_cast<int>(2), fk::maxValue<int2>),
                   (fk::make_set<double2>(fk::minValue<int>),
                    fk::make_set<double2>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<double2>(fk::maxValue<int>)),
                   fk::SaturateCast, int2, double2);

    ADD_UNARY_TEST((fk::minValue<int3>, fk::maxValue<int3> / static_cast<int>(2), fk::maxValue<int3>),
                   (fk::make_set<double3>(fk::minValue<int>),
                    fk::make_set<double3>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<double3>(fk::maxValue<int>)),
                   fk::SaturateCast, int3, double3);

    ADD_UNARY_TEST((fk::minValue<int4>, fk::maxValue<int4> / static_cast<int>(2), fk::maxValue<int4>),
                   (fk::make_set<double4>(fk::minValue<int>),
                    fk::make_set<double4>(fk::maxValue<int> / static_cast<int>(2)),
                    fk::make_set<double4>(fk::maxValue<int>)),
                   fk::SaturateCast, int4, double4);
}

void addCharIntTests() {

    // auto p = static_cast<char>(fk::maxValue<char1> / static_cast<char>(2));

    ADD_UNARY_TEST((fk::minValue<char1>, fk::make_set<char1>({static_cast<char>((fk::maxValue<char1> / 2).x)}),
                    fk::maxValue<char1>),
                   (fk::make_set<int1>(static_cast<int> (fk::minValue<char1>.x) / static_cast<int>(1)),
                    fk::make_set<int1>(static_cast<int>(fk::maxValue<char1>.x) / static_cast<int>(2)),
                    fk::make_set<int1>(static_cast<int>(fk::maxValue<char1>.x) / static_cast<int>(1))),
                   fk::SaturateCast, char1, int1);

    ADD_UNARY_TEST((fk::minValue<char2>, fk::make_set<char2>({static_cast<char>((fk::maxValue<char2> / 2).x)}),
                    fk::maxValue<char2>),
                   (fk::make_set<int2>(static_cast<int>(fk::minValue<char2>.x) / static_cast<int>(1)),
                    fk::make_set<int2>(static_cast<int>(fk::maxValue<char2>.x) / static_cast<int>(2)),
                    fk::make_set<int2>(static_cast<int>(fk::maxValue<char2>.x) / static_cast<int>(1))),
                   fk::SaturateCast, char2, int2);

    ADD_UNARY_TEST((fk::minValue<char3>, fk::make_set<char3>({static_cast<char>((fk::maxValue<char3> / 2).x)}),
                    fk::maxValue<char3>),
                   (fk::make_set<int3>(static_cast<int>(fk::minValue<char3>.x) / static_cast<int>(1)),
                    fk::make_set<int3>(static_cast<int>(fk::maxValue<char3>.x) / static_cast<int>(2)),
                    fk::make_set<int3>(static_cast<int>(fk::maxValue<char3>.x) / static_cast<int>(1))),
                   fk::SaturateCast, char3, int3);

    ADD_UNARY_TEST((fk::minValue<char4>, fk::make_set<char4>({static_cast<char>((fk::maxValue<char4> / 2).x)}),
                    fk::maxValue<char4>),
                   (fk::make_set<int4>(static_cast<int>(fk::minValue<char4>.x) / static_cast<int>(1)),
                    fk::make_set<int4>(static_cast<int>(fk::maxValue<char4>.x) / static_cast<int>(2)),
                    fk::make_set<int4>(static_cast<int>(fk::maxValue<char4>.x) / static_cast<int>(1))),
                   fk::SaturateCast, char4, int4);

}

START_ADDING_TESTS
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

// Int
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

// Char
addCharIntTests();
STOP_ADDING_TESTS

// You can add more tests for other type combinations as needed.
int launch() {

    RUN_ALL_TESTS
};