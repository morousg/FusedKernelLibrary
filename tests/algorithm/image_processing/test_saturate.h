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
#include <fused_kernel/core/utils/vlimits.h>
#include <tests/operation_test_utils.h>

// uint[1234]->uint[1234]
void addSintSintTests() {
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
// uint[1234]->int[1234]
void addUintSintTests() {
    ADD_UNARY_TEST((fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
                   (0, fk::maxValue<int1>, fk::maxValue<int1>), fk::SaturateCast, uint1, int1);

    ADD_UNARY_TEST((fk::minValue<uint2>, fk::maxValue<uint2> / static_cast<uint>(2), fk::maxValue<uint2>),
                   (fk::make_set<int2>(0), fk::maxValue<int2>, fk::maxValue<int2>), fk::SaturateCast, uint2, int2);

    ADD_UNARY_TEST((fk::minValue<uint3>, fk::maxValue<uint3> / static_cast<uint>(2), fk::maxValue<uint3>),
                   (fk::make_set<int3>(0), fk::maxValue<int3>, fk::maxValue<int3>), fk::SaturateCast, uint3, int3);

    ADD_UNARY_TEST((fk::minValue<uint4>, fk::maxValue<uint4> / static_cast<uint>(2), fk::maxValue<uint4>),
                   (fk::make_set<int4>(0), fk::maxValue<int4>, fk::maxValue<int4>), fk::SaturateCast, uint4, int4);
}

void addUintUCharTests() {
    ADD_UNARY_TEST((fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
                   (0, fk::maxValue<uchar1>, fk::maxValue<uchar1>), fk::SaturateCast, uint1, uchar1);

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

void addUintSCharTests() {

    ADD_UNARY_TEST((fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
                   (0, fk::maxValue<char1>, fk::maxValue<char1>), fk::SaturateCast, uint1, char1);

    ADD_UNARY_TEST((fk::minValue<uint2>, fk::maxValue<uint2> / static_cast<uint>(2), fk::maxValue<uint2>),
                   (fk::make_set<char2>(0), fk::maxValue<char2>, fk::maxValue<char2>), fk::SaturateCast, uint2, char2);

    ADD_UNARY_TEST((fk::minValue<uint3>, fk::maxValue<uint3> / static_cast<uint>(2), fk::maxValue<uint3>),
                   (fk::make_set<char3>(0), fk::maxValue<char3>, fk::maxValue<char3>), fk::SaturateCast, uint3, char3);

    ADD_UNARY_TEST((fk::minValue<uint4>, fk::maxValue<uint4> / static_cast<uint>(2), fk::maxValue<uint4>),
                   (fk::make_set<char4>(0), fk::maxValue<char4>, fk::maxValue<char4>), fk::SaturateCast, uint4, char4);
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


void addUintSLongTests() {

    ADD_UNARY_TEST(
        (fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
        (fk::make_set<long1>(0), fk::maxValue<uint> / static_cast<uint>(2), fk::make_set<long1>(fk::maxValue<uint>)),
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

void addUintSLongLongTests() {

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


START_ADDING_TESTS
// uint
addUintUintTests();
addUintSintTests();
addUintUCharTests();
addUintSCharTests();
addUintULongTests();
addUintULongLongTests();

addSintSintTests();
/*
// double[1234]->double[1234]
ADD_UNARY_TEST((fk::minValue<double1>, fk::maxValue<double1> / static_cast<double>(2.f), fk::maxValue<double1>),
           (fk::minValue<double1>, fk::maxValue<double1> / static_cast<double>(2.f), fk::maxValue<double1>),
           fk::SaturateCast, double1, double1)

// auto x = fk::minValue<double1>;
// double f = std::numeric_limits<double>::min();
// auto dmin = DBL_MIN;

ADD_UNARY_TEST((fk::minValue<double2>, fk::make_set<double2>(-200.6), fk::make_set<double2>(200.6),
            fk::maxValue<double2>),
           (fk::minValue<double2>, fk::make_set<double2>(-200.6f), fk::make_set<double2>(200.6f),
            fk::maxValue<double2>),
           fk::SaturateCast, double2, float2)

ADD_UNARY_TEST((fk::minValue<double2>, fk::make_set<double2>(-200.6), fk::make_set<double2>(200.6),
            fk::maxValue<double2>),
           (fk::minValue<float2>, fk::make_set<float2>(-200.6f), fk::make_set<float2>(200.6f),
            fk::maxValue<float2>),
           fk::SaturateCast, double2, float2)

ADD_UNARY_TEST((fk::minValue<double2>, fk::make_set<double2>(-200.6), fk::make_set<double2>(200.6),
            fk::maxValue<double2>),
           (fk::minValue<float2>, fk::make_set<float2>(-200.6f), fk::make_set<float2>(200.6f),
            fk::maxValue<float2>),
           fk::SaturateCast, double2, float2)
*/
STOP_ADDING_TESTS

// You can add more tests for other type combinations as needed.
int launch() {

    RUN_ALL_TESTS
};