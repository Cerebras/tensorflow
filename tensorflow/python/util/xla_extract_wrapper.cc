/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include "include/pybind11/pybind11.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/tools/xla_extract/tf_graph_to_xla_lib.h"

namespace py = pybind11;

struct TF_Status;

PYBIND11_MODULE(_pywrap_xla_extract, m) {
  m.def("ExtractXlaWithStringInputs",
        [](const std::string* graph_def_string,
           const std::string* targets_string) {
               std::string result;
               tensorflow::Status extraction_status =
                   tensorflow::xla_extract_via_strings(*graph_def_string,
                                                       *targets_string,
                                                       &result);
               MaybeRaiseFromStatus(extraction_status);
               return result;
        });
}
