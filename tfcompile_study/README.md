### tf_compile

Note: based of tensorflow r2.0 branch   (to build we need to update git > 2.0)
  * branch used can be found at https://github.com/Cerebras/tensorflow/tree/vishal/tfcompile
  *  this tf branch was build using the default settings in configure.

Only change in the code base is in `tensorflow/compiler/aot/codegen.cc#L774` to support `/` in node names.  
Originally it supports names that follow C++11 Standard naming convention, so this change to handle variable_scope.

Build tfcompile:
  ```bash
  bazel --output_user_root=/spare/vish_new build -c opt  //tensorflow/compiler/aot:tfcompile
  ```

`tf_compile` command line tool requires atleast
  * graph (.pbtxt or .pb)
  * config (.pbtxt or .pb)
  * cpp_class (for generated .h  and .o files)


graph is extracted from the model (setting `add_shapes=True`, did not require `xla.compile`)  
cpp_class is set the same as in their examples (`mynamespace::MyComputation`)

config:
  * This contains `feed` (input) nodes, `fetch` (output) nodes and `variable` nodes.
  * The examples in the Tensorflow repo, are all generated manually, but for simple graphs, we can generate the config file based of the graph
  * the config is based on the proto file (`tensorflow/compiler/tf2xla/tf2xla.proto`)



To compile protobuf file  
```bash
protoc --python_out=/path_to_store_compiled_file --proto_path=/path_to_tensorflow_dir/tensorflow   tensorflow/compiler/tf2xla/tf2xla.proto
```

To run tfcompile:  
```bash
TF_CPP_MIN_VLOG_LEVEL=3 /path_to_tensorflow/tensorflow/bazel-bin/tensorflow/compiler/aot/tfcompile --graph=graph_modelfn.pbtxt --config=config_modelfn.config.pbtxt --cpp_class="mynamespace::MyComputation"
```


#### Experiments:
1. Experiments which passed: (able to compile successfully and viewed xla thorugh env variable (`TF_CPP_MIN_VLOG_LEVEL=3`))
  * fully connected layers with `GradientDescentOptimizer`
  * fully connected layers with `AdamOptimizer`
  * fully connected layers with `batch_normalization(training=True)` and `GradientDescentOptimizer`
  * fully connected layers with `batch_normalization(training=True)` and `AdamOptimizer`
  * cnn + fc layers with `GradientDescentOptimizer`
  * cnn + fc layers with `AdamOptimizer`
  * cnn + fc layers with `batch_normalization(training=True)` and `GradientDescentOptimizer`
  * cnn + fc layers with `batch_normalization(training=True)` and `AdamOptimizer`

2. Experiments which failed:
  * `BasicRNNCell` with `dynamic_rnn` - ```INVALID ARGUMENTS: XLA compilation requires a fixed stack size upper bound. If you are using tf.while_loop, set the maximum_iterations parameter to fix this issue.
	 [[{{node dynamic_rnn/gradients/dynamic_rnn/func/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc}}]]```
 * Using `tf.keras.layers.SimpleRNNCell` and `tf.keras.layers.RNN` - same error as above
 * fully connected layers with `batch_normalization(training=False)` and `GradientDescentOptimizer`- ```Non-OK-status: status status: Internal: RET_CHECK failure (tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc:32) ShapeUtil::IndexIsValid(alias_.shape(), output_index) Tring to set up alias at {9} which is an invalid index for shape (f32[], f32[2,2,1,1], f32[1], f32[729,256], f32[256], f32[256], f32[256], f32[256,10], f32[10])```
 * cnn + fc layers with `batch_normalization(training=False)` and `GradientDescentOptimizer` - same error as above  
    * In master branch, they are adding a read_only option to the variable config, which should solve this issue. Modified function to accept this has been added to the utils.py

Major concerns/Feedback:
 * generating config file manually isn't feasible - script to generate this is in utils.py (for r2.0 and master branch versions)
 * naming of variables is extremely constrained, if they can loosen it for the xla generation, and can maintain the constraints for the output file generation that will be ideal.
 * rnn - some reason tf.while_loop isnt using maximum_iterations though its specified. (this feels like a red herring, seems to be the same issue as we had with our extractor, of using device xla_cpu_jit instead of xla_cpu)
 * `keras.layers.BatchNormalization` uses batch_normalization_v1 which doesn't seem to add the `update_ops` as the `tf.layers.batch_normalization`. - in general keras based layers/ops tend to use older version of variables and underneath functions.

Files:  
* utils.py - contains config generation script and run script to generate graph_def and config script from model_fn and input_fn
* tf2xla_pb2.py - compiled protobuf file for python import
* examples contain the different models tried.
