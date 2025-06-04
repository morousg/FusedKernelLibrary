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

#ifndef FK_REF_CLASS_H
#define FK_REF_CLASS_H

#include <stdexcept>

namespace fk {

    class Ref {
        struct RefPtr {
            int cnt{ 0 };
        };
        RefPtr* ref{ nullptr };
        inline void initFromOther(const Ref& other) {
            if (other.ref) {
                ref = other.ref;
                ref->cnt++;
            }
        }
    public:
        inline Ref() {
            ref = new RefPtr;
            if (ref != nullptr) {
                ref->cnt = 1;
            } else {
                throw std::runtime_error("Failed to allocate memory for reference counter.");
            }
        }

        inline Ref(const Ref& other) {
            initFromOther(other);
        }

        Ref(Ref&&) = delete;
        Ref& operator=(Ref&&) = delete;
        
        Ref& operator=(const Ref& other) {
            if (this != &other) {
                initFromOther(other);
            }
            return *this;
        }

        virtual inline ~Ref() {
            if (ref) {
                ref->cnt--;
                if (ref->cnt == 0) {
                    delete ref;
                }
            }
        }
        inline int getRefCount() const {
            return ref ? ref->cnt : 0;
        }

    };

} // namespace fk

#endif // !FK_REF_CLASS_H
