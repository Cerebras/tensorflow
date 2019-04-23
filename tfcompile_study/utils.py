from tf2xla_pb2 import Config, TensorId


def config_from_graph(input_tensors,
                      output_tensors,
                      graph_def,
                      is_training=None):
    """
    Generate config based on graphdef
    Args:
        input_tensors: inputs for the model function (eg. features, labels etc.)
        output_tensors: outputs from the model function (eg. loss)
        graph_def: GraphDef of the model.
        is_training: if None, uses r2.0 version of tensorflow
                    if Bool uses commit 0886c6e0736135fdc4bbb4905b88158b66955abe version (master at that time)
                    of tensorflow which has the read_only flag
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
            if is_training is None:
                config.variable.add(
                    node_name=graph_def.node[i].name,
                    shape=graph_def.node[i].attr["shape"].shape,
                    type=graph_def.node[i].attr["dtype"].type)
            else:
                config.variable.add(
                    node_name=graph_def.node[i].name,
                    shape=graph_def.node[i].attr["shape"].shape,
                    type=graph_def.node[i].attr["dtype"].type,
                    read_only=not is_training)

    return config


def run(model_fn, input_fn, file_name):
    x, y = input_fn()
    out = model_fn(x, y)
    graph = out.graph.as_graph_def(add_shapes=True)
    file_graph = "graph_" + file_name + ".pbtxt"
    with open(file_graph, 'w') as f:
        f.write(str(graph))
    config = config_from_graph([x, y], [out], graph)
    file_config = "config_" + file_name + ".config.pbtxt"
    with open(file_config, 'w') as f:
        f.write(str(config))