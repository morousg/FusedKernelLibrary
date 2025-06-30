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

#ifndef FK_TYPE_TO_STRING_H
#define FK_TYPE_TO_STRING_H

#include <string>
#include <typeinfo>
#if !defined(__NVCC__) && !defined(NVRTC_COMPILER)
#include <string_view>
namespace fk {
    constexpr inline std::string_view getTypeNamePretty_helper(const std::string_view& signature,
                                                               const std::string_view& prefix,
                                                               const std::string_view& suffix) {
        auto start = signature.find(prefix);
        if (start == std::string_view::npos) {
            return "unknown (prefix not found)";
        } else {
            start += prefix.length();
        }

        auto end = signature.find(suffix, start);
        if (end == std::string_view::npos) {
            return "unknown (suffix not found)";
        }

        return signature.substr(start, end - start);
    }

    template <typename T>
    constexpr inline std::string_view typeToStringView() {
#if defined(__clang__)
        constexpr std::string_view signature = __PRETTY_FUNCTION__;
        constexpr std::string_view prefix = "[T = ";
        constexpr std::string_view suffix = "]";
        return getTypeNamePretty_helper(signature, prefix, suffix);
#elif defined(__GNUC__)
        constexpr std::string_view signature = __PRETTY_FUNCTION__;
        constexpr std::string_view prefix = "[with T = "; // Common case
        constexpr std::string_view suffix = ";";
        return getTypeNamePretty_helper(signature, prefix, suffix);
#elif defined(_MSC_VER)
        constexpr std::string_view signature{ __FUNCSIG__ };
        constexpr std::string_view prefix{ "fk::typeToStringView<" };
        constexpr std::string_view suffix{ ">()" };
        return getTypeNamePretty_helper(signature, prefix, suffix);
#else
        return "unknown (unsupported compiler)";
#endif // __clang__ __GNUC__ _MSC_VER
    }
} // namespace fk
#endif // __NVCC__ __CUDACC__

#include <vector>
#include <map>
#include <typeinfo> // Required for typeid

// For GCC and Clang, include cxxabi.h for __cxa_demangle
// The condition `!defined(_MSC_VER)` is to ensure this path is not taken by Clang-cl,
// which should ideally follow the MSVC path for typeid().name() behavior.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#include <cxxabi.h>
#include <cstdlib> // For std::free
#include <memory>  // For std::unique_ptr (though not strictly necessary here, good practice for RAII if used)
namespace fk {
    namespace detail {
        // Demangles a C++ name using abi::__cxa_demangle
        std::string demangle_name(const char* mangled_name) {
            int status = 0;
            // abi::__cxa_demangle allocates memory using malloc, which must be freed by std::free
            char* demangled_c_str = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);

            if (status == 0 && demangled_c_str) {
                std::string demangled_name(demangled_c_str);
                std::free(demangled_c_str);
                return demangled_name;
            }
            // If demangling fails or is not needed, return the original (likely mangled) name
            return mangled_name;
        }
    } // namespace detail

    template <typename T>
    std::string typeToString() {
        return detail::demangle_name(typeid(T).name());
    }
} // namespace fk
#elif defined(_MSC_VER)
// For MSVC (and Clang-cl, as it defines _MSC_VER), typeid(T).name() is often already
// human-readable but may include prefixes like "class ", "struct ", "enum ".
namespace fk {
    namespace detail {
        // Helper to remove common prefixes from MSVC's typeid(T).name() output.
        std::string clean_msvc_typename(std::string name) {
            // List of prefixes to remove
            const std::string prefixes[] = { "class ", "struct ", "enum " };

            for (const auto& prefix : prefixes) {
                if (name.length() >= prefix.length() && name.substr(0, prefix.length()) == prefix) {
                    name.erase(0, prefix.length());
                    // It's unlikely for these prefixes to be chained in a way that requires re-looping
                    // for this specific set (e.g., "class struct X" is not typical output).
                    break;
                }
            }
            return name;
        }
    } // namespace detail

    template <typename T>
    std::string typeToString() {
        return detail::clean_msvc_typename(typeid(T).name());
    }
} // namespace fk
#else
// Fallback for other compilers: return the raw name from typeid.
// This might be mangled or not, depending on the compiler.
namespace fk {
    template <typename T>
    std::string typeToString() {
        return typeid(T).name();
    }
} // namespace fk
#endif

#endif // FK_TYPE_TO_STRING_H
