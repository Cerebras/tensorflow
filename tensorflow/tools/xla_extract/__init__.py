# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""python interface function to extract xla for a given target_op's graph"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.core.framework import graph_pb2
from tensorflow.compiler.xla.service import hlo_pb2
from tensorflow.python.framework import errors
from tensorflow.python.pywrap_tensorflow import ExtractXlaWithStringInputs
from tensorflow.python.util import compat


def XlaExtract(target_op):
  """Python wrapper for the XLA extraction tool
  Args:
  op with graph to be compiled to xla hlo
  Returns:
  New Xla HloModuleProto
  """
  targets_string = compat.as_bytes(target_op.name)
  graph_def_string = target_op.graph.as_graph_def(
      add_shapes=True).SerializeToString()

  with errors.raise_exception_on_not_ok_status() as status:
    hlo_mod_string = ExtractXlaWithStringInputs(
        graph_def_string, targets_string, status)
    hlo_snapshot_def = hlo_pb2.HloModuleProto()
    hlo_snapshot_def.ParseFromString(hlo_mod_string)
    return hlo_snapshot_def
