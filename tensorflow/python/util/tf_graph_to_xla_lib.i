%module tf_graph_to_xla_lib
%include <std_string.i>
%include "tensorflow/python/lib/core/strings.i"
%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/tools/xla_extract/tf_graph_to_xla_lib.h"
%}
%ignoreall
%unignore tensorflow;
%unignore ExtractXlaWithStringInputs;
%{
string ExtractXlaWithStringInputs(string graph_def_string,
				  string targets_string,
				  TF_Status* out_status) {
  string result;
  tensorflow::Status extraction_status =
    tensorflow::xla_extract_via_strings(graph_def_string,
					targets_string,
					&result);

  Set_TF_Status_from_Status(out_status, extraction_status);
  return result;
}

%}
#include "tensorflow/tools/xla_extract/tf_graph_to_xla_lib.h"
string ExtractXlaWithStringInputs(string graph_def_string,
				  string targets_string,
				  TF_Status* out_status);
%unignoreall

