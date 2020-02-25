/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_UTIL_UTIL_H_
#define TENSORFLOW_CORE_UTIL_UTIL_H_

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

// If op_name has '/' in it, then return everything before the first '/'.
// Otherwise return empty string.
StringPiece NodeNamePrefix(const StringPiece& op_name);

// If op_name has '/' in it, then return everything before the last '/'.
// Otherwise return empty string.
StringPiece NodeNameFullPrefix(const StringPiece& op_name);

class MovingAverage {
 public:
  explicit MovingAverage(int window);
  ~MovingAverage();

  void Clear();

  double GetAverage() const;
  void AddValue(double v);

 private:
  const int window_;  // Max size of interval
  double sum_;        // Sum over interval
  double* data_;      // Actual data values
  int head_;          // Offset of the newest statistic in data_
  int count_;         // # of valid data elements in window
};

// Returns a string printing bytes in ptr[0..n).  The output looks
// like "00 01 ef cd cd ef".
string PrintMemory(const char* ptr, size_t n);

// Given a flattened index into a tensor, computes a string s so that
// StrAppend("tensor", s) is a Python indexing expression.  E.g.,
// "tensor", "tensor[i]", "tensor[i, j]", etc.
string SliceDebugString(const TensorShape& shape, const int64 flat);

// disable MKL in runtime
#ifdef INTEL_MKL
bool DisableMKL();
#endif  // INTEL_MKL

}  // namespace tensorflow


#include <sys/syscall.h>
#include <zconf.h>

class EnterLeave {
    static __thread int depth_;
    static const std::string library_;
    const std::string label_;
    const pid_t thread_id_;
public:
    static std::string concat(const char *s0, const char *s1, const char *s2) {
      std::string s;
      if (s0 && *s0) {
        s = s0;
        s += "::";
      }
      s += s1;
      s += " (";
      s += s2;
      s += ")";
      return s;
    }
    inline EnterLeave(const std::string label) : label_(label), thread_id_(syscall(SYS_gettid)) {
      for (int x = 0; x < depth_; ++x) {
        printf("  ");
      }
      printf("ENTER[%d (%s)]: %s\n", thread_id_, library_.c_str(), label.c_str());
      fflush(stdout);
      ++depth_;
    }
    inline ~EnterLeave() {
      --depth_;
      for (int x = 0; x < depth_; ++x) {
        printf("  ");
      }
      printf("LEAVE[%d (%s)]: %s\n", thread_id_, library_.c_str(), label_.c_str());
      fflush(stdout);
    }
};

#define HERE() EnterLeave __here(EnterLeave::concat(nullptr, __PRETTY_FUNCTION__, __FILE__))

#include "external/protobuf_archive/src/google/protobuf/util/json_util.h"

template <typename MSG>
inline std::string m2j(const MSG& msg) {
  std::string json;
  google::protobuf::util::JsonPrintOptions op;
  op.add_whitespace = true;
  google::protobuf::util::MessageToJsonString(msg, &json, op);
  return std::move(json);
}

/**
 * Convenient endl with flush for debugging
 */
#define ENDL std::endl << std::flush

#endif  // TENSORFLOW_CORE_UTIL_UTIL_H_
