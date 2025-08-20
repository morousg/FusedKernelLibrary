/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Hguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "tests/main.h"

#include <fused_kernel/core/data/string.h>
#include <sstream>

int launch() {
    bool result{ true };
    {
        fk::String str1("");
        fk::String str2("Hello");
        fk::String str3(" ");
        fk::String str4("World!");

        std::stringstream ss;

        ss << str1 << str2 << str3 << str4;
        if (ss.str() != std::string("Hello World!")) {
            std::cout << "String operator<< failed: result " << ss.str() <<
                " expected: " << "Hello World!" << std::endl;
            result &= false;
        }
    }

    {
        fk::String str1("");
        fk::String str2("Hello");
        fk::String str4("World!");

        auto str5 = str1 + " Hi " + str2 + " brave " + " new " + str4;

        if (!(str5 == fk::String(" Hi Hello brave  new World!"))) {
            std::cout << "String operator+ with const char* elements failed" << std::endl;
            result &= false;
        }
    }

    {
        constexpr fk::String str1("");
        constexpr fk::String str2("Hello");
        constexpr fk::String str3(" ");
        constexpr fk::String str4("World!");
        constexpr fk::String str5 = str1 + str2 + str3 + str4;

        static_assert(str5 == fk::String("Hello World!"), "Error in operator== in constexpr context");
        result &= true;
    }

    return result ? 0 : -1;
}