from tf2xla_pb2 import Config, TensorId
import tensorflow as tf
import subprocess

def config_from_graph(input_tensors,
                      output_tensors,
                      graph_def,
                      trainable_vars_names,
                      is_training):
    """
    Generate config based on graphdef
    Args:
        input_tensors: inputs for the model function (eg. features, labels etc.)
        output_tensors: outputs from the model function (eg. loss)
        graph_def: GraphDef of the model.
        trainable_vars_names: trainable variable names from according to tensorflow graph
        is_training: If Batchnorm/Dropout training is False
    Returns config proto for tfcompile
    """
    config = Config()
    num_nodes = len(graph_def.node)
    input_tensor_names = [tensor.op.name for tensor in input_tensors]
    output_tensor_names = [tensor.op.name for tensor in output_tensors]
    for i in range(num_nodes):
        if graph_def.node[i].name in input_tensor_names:
            tensor_id = TensorId(node_name=graph_def.node[i].name)
            config.feed.add(id=tensor_id,
                            shape=graph_def.node[i].attr["shape"].shape,
                            type=graph_def.node[i].attr["dtype"].type)

        if graph_def.node[i].name in output_tensor_names:
            tensor_id = TensorId(node_name=graph_def.node[i].name)
            config.fetch.add(id=tensor_id)

        if graph_def.node[i].op == "VarHandleOp":
            if graph_def.node[i].name in trainable_vars_names:
                config.variable.add(
                    node_name=graph_def.node[i].name,
                    shape=graph_def.node[i].attr["shape"].shape,
                    type=graph_def.node[i].attr["dtype"].type,
                    readonly=False)
            else:
                config.variable.add(
                    node_name=graph_def.node[i].name,
                    shape=graph_def.node[i].attr["shape"].shape,
                    type=graph_def.node[i].attr["dtype"].type,
                    readonly=not is_training)

    return config



def run(model_fn, input_fn, file_name, is_training=True, only_gen=False):
    """
    From model_fn and input_fn to generating tfcompile inputs and outputs
    Args:
        model_fn: model definition which takes the outputs of the input_fn as inputs
        input_fn: placeholders of the inputs needed for the model_fn
        file_name: naming of model for generated files (inputs to tfcompile) .
        only_gen: If we want only the inputs to be generated and not the outputs of tfcompile
        is_training: For operations like batchnorm, to capture which variables are being trained and which ones won't be
    """
    x,y = input_fn()
    out = model_fn(x,y)
    trainable_vars_names =[var.op.name for var in tf.trainable_variables(scope=None)]
    graph = out.graph.as_graph_def(add_shapes=True)
    file_graph = "graph_" + file_name + ".pbtxt"
    with open(file_graph, 'w') as f:
        f.write(str(graph))
    config = config_from_graph([x, y], [out], graph, trainable_vars_names, is_training)
    file_config = "config_" + file_name + ".config.pbtxt"
    with open(file_config, 'w') as f:
        f.write(str(config))
    if not only_gen:
        subprocess.run([
        "./../../../bazel-bin/tensorflow/compiler/aot/tfcompile",
        "--graph=" + file_graph, "--config=" + file_config,
        "--cpp_class=mynamespace::MyComputation"])