from tf2xla_pb2 import Config, TensorId
import tensorflow as tf
import subprocess


def config_from_graph(input_tensors, output_tensors, graph_def,
                      trainable_vars_names, is_training):
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
    input_tensor_names = [tensor.op.name for tensor in input_tensors]
    output_tensor_names = [tensor.op.name for tensor in output_tensors]

    def _shape_type(n):
        return {"shape": n.attr["shape"].shape, "type": n.attr["dtype"].type}

    for node in graph_def.node:
        if node.name in input_tensor_names:
            config.feed.add(id=TensorId(node_name=node.name),
                            **_shape_type(node))

        if node.name in output_tensor_names:
            config.fetch.add(id=TensorId(node_name=node.name))

        if node.op == "VarHandleOp":
            trainable = node.name in trainable_vars_names
            read_write = trainable or is_training
            var_args = {
                "node_name": node.name,
                "readonly": not read_write,
                **_shape_type(node),
            }
            config.variable.add(**var_args)

    return config


def run(model_fn, inputs, file_name, is_training=True, only_gen=False):
    """
    From model_fn and input_fn to generating tfcompile inputs and outputs
    Args:
        model_fn: model definition which takes the outputs of the input_fn as inputs
        input_fn: placeholders of the inputs needed for the model_fn
        file_name: naming of model for generated files (inputs to tfcompile) .
        only_gen: If we want only the inputs to be generated and not the outputs of tfcompile
        is_training: For operations like batchnorm, to capture which variables are being trained and which ones won't be
    """
    out = model_fn(*inputs, is_training)
    trainable_vars_names = [var.op.name for var in tf.trainable_variables()]
    graph = out.graph.as_graph_def(add_shapes=True)
    file_graph = "graph_" + file_name + ".pbtxt"
    with open(file_graph, 'w') as f:
        f.write(str(graph))
    config = config_from_graph(inputs, [out], graph, trainable_vars_names,
                               is_training)
    file_config = "config_" + file_name + ".config.pbtxt"
    with open(file_config, 'w') as f:
        f.write(str(config))
    if not only_gen:
        output = subprocess.run([
            "tfcompile", "--graph=" + file_graph, "--config=" + file_config,
            "--cpp_class=mynamespace::MyComputation"
        ])
        output.check_returncode()