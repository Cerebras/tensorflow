# XLA extraction from tensorflow graph

## Building
After configuring, use the following command to build:
```bash
bazel build tensorflow/tools/xla_extract:tf_graph_to_xla
```


## Usage
First generate the Tensorflow `GraphDef` object in text format protobuf. See script `tf_graph_example.py` for example that generates GraphDef source.

```bash
./tf_graph_example.py
```

You should get the following output:

```bash
2018-12-04 15:42:21.150934: I tensorflow/tools/graph_transforms/transform_graph.cc:317] Applying strip_unused_nodes
Target node: output_0
```

Now run the tool as follows:

```bash
tf_graph_to_xla --in_graph="tf_graph.pbtxt" --out_graph="xla_module.pbtxt" --target_node="output_0:0"
```

This will generate the xla module as a text format protobuf into `xla_module.pbtxt`.
