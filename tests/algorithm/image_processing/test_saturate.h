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
#include <fused_kernel/core/utils/vlimits.h>

#include <tests/operation_test_utils.h>

START_ADDING_TESTS
ADD_UNARY_TEST((0,200, 100), (0, 200, 100), fk::SaturateCast, uint, uint)
ADD_UNARY_TEST((fk::minValue<double2>, fk::make_set<double2>(-200.6), fk::make_set<double2>(200.6), fk::maxValue<double2>), 
               (fk::minValue<float2>, fk::make_set<float2>(-200.6f), fk::make_set<float2>(200.6f), fk::maxValue<float2>), fk::SaturateCast, double2, float2)
ADD_UNARY_TEST((fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>),
               (fk::minValue<uint1>, fk::maxValue<uint1> / static_cast<uint>(2), fk::maxValue<uint1>), fk::SaturateCast, uint1, uint1)
STOP_ADDING_TESTS

// You can add more tests for other type combinations as needed.
int launch() {
    RUN_ALL_TESTS
};