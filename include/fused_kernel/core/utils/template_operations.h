/* Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_TEMPLATE_OPERATIONS
#define FK_TEMPLATE_OPERATIONS

#include <utility>

namespace fk {
    template <bool... results>
    constexpr bool and_v = (results && ...);

    template <bool... results>
    constexpr bool or_v = (results || ...);

    template <typename T, T Element>
    struct Find {
    private:
        template <T Head>
        FK_HOST_DEVICE_FUSE
        int in(std::integer_sequence<T, Head>) {
            if constexpr (Head == Element) {
                return 0;
            } else {
                return -1;
            }
        }
    public:
        template <T Head, T... Tail>
        FK_HOST_DEVICE_FUSE
        int in(std::integer_sequence<T, Head, Tail...>) {
            if constexpr (Head == Element) {
                return 0;
            } else {
                constexpr int result = in(std::integer_sequence<T, Tail...>{});
                return result == -1 ? -1 : 1 + result;
            }
        }

        template <T... Sequence>
        FK_HOST_DEVICE_FUSE
        bool one_of(std::integer_sequence<T, Sequence...>) {
            if constexpr (in(std::integer_sequence<T, Sequence...>{}) == -1) {
                return false;
            } else {
                return true;
            }
        }
    };

    template <typename T, T Start, T... Ints>
    constexpr inline auto make_integer_sequence_from(const std::integer_sequence<T, Ints...>&) {
        return std::integer_sequence<T, (Start + Ints)...>{};
    }
} // namespace fk

#endif
