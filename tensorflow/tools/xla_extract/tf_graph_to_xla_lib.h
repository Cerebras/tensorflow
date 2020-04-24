#ifndef TENSORFLOW_CONTRIB_TF_GRAPH_TO_XLA_LIB_H
#define TENSORFLOW_CONTRIB_TF_GRAPH_TO_XLA_LIB_H

#include <stdio.h>
#include <unistd.h>
#include <iterator>
#include <string>
#include <tuple>
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/graph_execution_state.h" // ClientGraph
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/compiler/xla/service/service.h"

namespace tensorflow {

xla::HloModuleProto ExtractHloFromGraphDef(GraphDef&& in_graph,
                                           const std::string& fetch);

Status xla_extract_via_strings(const string& graph_def_msg,
                               const string& target_node, string* out_graph);
}  // namespace tensorflow

#endif
