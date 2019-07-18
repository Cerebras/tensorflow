from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 # pylint: disable=unused-import,wildcard-import, line-too-long
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