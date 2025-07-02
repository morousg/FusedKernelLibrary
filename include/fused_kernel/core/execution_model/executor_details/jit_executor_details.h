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

#ifndef FK_JIT_EXECUTOR_DETAILS_H
#define FK_JIT_EXECUTOR_DETAILS_H

#include <string>
#include <vector>
#include <cstring>

#if defined(NVRTC_ENABLED) || defined(ENABLE_CPU_JIT)
namespace fk {
    // --- Abstract Operation Definition (Hybrid C++ class) ---
    class JIT_Operation_pp {
    private:
        std::string opType; // The C++ typename of the operation struct
        void* opData;       // A pointer to an internal copy of the data (owned)
        size_t dataSize;    // The size of the data block

    public:
        // Constructor: Performs a deep copy of the provided data.
        JIT_Operation_pp(std::string type, const void* data, size_t size)
            : opType(type), dataSize(size) { // Changed std::move(type) to type
            // Allocate memory and copy the parameter data
            opData = new char[dataSize];
            memcpy(opData, data, dataSize);
        }

        // Copy Constructor: Essential for use in std::vector.
        JIT_Operation_pp(const JIT_Operation_pp& other)
            : opType(other.opType), dataSize(other.dataSize) {
            // Allocate and copy data for the new object
            opData = new char[dataSize];
            memcpy(opData, other.opData, dataSize);
        }

        // Move Constructor
        JIT_Operation_pp(JIT_Operation_pp&& other) noexcept
            : opType(std::move(other.opType)), opData(other.opData), dataSize(other.dataSize) {
            // Take ownership of the other object's resources
            other.opData = nullptr;
            other.dataSize = 0;
        }

        // Copy Assignment Operator
        JIT_Operation_pp& operator=(const JIT_Operation_pp& other) {
            if (this == &other) {
                return *this;
            }
            // Free old resources
            delete[] static_cast<char*>(opData);

            // Copy new resources
            opType = other.opType;
            dataSize = other.dataSize;
            opData = new char[dataSize];
            memcpy(opData, other.opData, dataSize);

            return *this;
        }

        // Move Assignment Operator
        JIT_Operation_pp& operator=(JIT_Operation_pp&& other) noexcept {
            if (this == &other) {
                return *this;
            }
            delete[] static_cast<char*>(opData);

            opType = std::move(other.opType);
            opData = other.opData;
            dataSize = other.dataSize;

            other.opData = nullptr;
            other.dataSize = 0;

            return *this;
        }


        // Destructor: Frees the owned memory using RAII.
        ~JIT_Operation_pp() {
            // Cast to char* to ensure correct byte-wise deletion with delete[]
            delete[] static_cast<char*>(opData);
        }

        // Public accessors
        const std::string& getType() const { return opType; }
        void* getData() const { return opData; }
    };

#if defined(NVRTC_ENABLED)
    // NVRTC-specific code remains here
    namespace jit_internal {
        // --- Helper Functions for Dynamic Pipeline Construction ---
        std::string buildNameExpression(const std::string& kernelName, const std::vector<JIT_Operation_pp>& pipeline);
        std::vector<void*> buildKernelArguments(const std::vector<JIT_Operation_pp>& pipeline);

        template <typename... IOps>
        std::vector<JIT_Operation_pp> buildOperationPipeline(const IOps&... iOps) {
            std::vector<JIT_Operation_pp> pipeline;
            (pipeline.emplace_back(typeToString<IOps>(), &iOps, sizeof(IOps)), ...);
            return pipeline;
        }
    } // jit_internal

    // Other NVRTC-specific classes and functions would go here...
#endif // NVRTC_ENABLED

} // namespace fk
#endif // NVRTC_ENABLED || ENABLE_CPU_JIT

#endif // FK_JIT_EXECUTOR_DETAILS_H