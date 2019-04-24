### tf_compile

Note: based of tensorflow r2.0 branch (to build we need to update git > 2.0)
  * branch used can be found at https://github.com/Cerebras/tensorflow/tree/vishal/tfcompile
  *  this tf branch was build using the default settings in configure.

Only change in the code base is in `tensorflow/compiler/aot/codegen.cc#L774` to support `/` in node names.  
Originally it supports names that follow C++11 Standard naming convention, so this change to handle variable_scope.

Build tfcompile:
  ```bash
  bazel --output_user_root=/spare/build_new build -c opt  //tensorflow/compiler/aot:tfcompile
  ```

`tf_compile` command line tool requires at least:
  * graph (.pbtxt or .pb)
  * config (.pbtxt or .pb)
  * cpp_class (for generated .h  and .o files)


graph is extracted from the model (setting `add_shapes=True`, did not require `xla.compile`)  
cpp_class is set the same as in their examples (`mynamespace::MyComputation`)

config:
  * This contains `feed` (input) nodes, `fetch` (output) nodes and `variable` nodes.
  * The examples in the Tensorflow repository, are all generated manually, but for simple graphs, we can generate the config file based of the graph
  * the config is based on the proto file (`tensorflow/compiler/tf2xla/tf2xla.proto`)



To compile protobuf file  
```bash
protoc --python_out=/path_to_store_compiled_file --proto_path=/path_to_tensorflow_dir/tensorflow   tensorflow/compiler/tf2xla/tf2xla.proto
```

To run tfcompile:  
```bash
TF_CPP_MIN_VLOG_LEVEL=3 /path_to_tensorflow/tensorflow/bazel-bin/tensorflow/compiler/aot/tfcompile --graph=graph_model_fn.pbtxt --config=config_model_fn.config.pbtxt --cpp_class="mynamespace::MyComputation"
```


#### Experiments:
1. Experiments which passed: (able to compile successfully and viewed xla through env variable (`TF_CPP_MIN_VLOG_LEVEL=3`))
  * fully connected layers with `GradientDescentOptimizer`
  * fully connected layers with `AdamOptimizer`
  * fully connected layers with `batch_normalization(training=True)` and `GradientDescentOptimizer`
  * fully connected layers with `batch_normalization(training=True)` and `AdamOptimizer`
  * fully connected layers with `keras.BatchNormalization()` works if updates are captured and added (based of Min's PR (#17746) for Xception)
  * cnn + fc layers with `GradientDescentOptimizer`
  * cnn + fc layers with `AdamOptimizer`
  * cnn + fc layers with `batch_normalization(training=True)` and `GradientDescentOptimizer`
  * cnn + fc layers with `batch_normalization(training=True)` and `AdamOptimizer`

2. Experiments which failed:
  * `BasicRNNCell` with `dynamic_rnn` - ```INVALID ARGUMENTS: XLA compilation requires a fixed stack size upper bound. If you are using tf.while_loop, set the maximum_iterations parameter to fix this issue.
	 [[{{node dynamic_rnn/gradients/dynamic_rnn/func/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc}}]]```
      * Both with and without specifying sequence_length failed.
      *  In our extract tool, if sequence_length is None, it can extract xla successfully.
      *  While setting up our extract tool, rnns would generate two function defs (my understanding was one was xla_cpu, and other was xla_cpu_jit specific), the xla_cpu_jit specific version would give a similar error as above (since tfcompile uses xla_cpu_jit as device, I think its the same issue.)
 * Using `tf.keras.layers.SimpleRNNCell` and `tf.keras.layers.RNN` - same error as above
 * fully connected layers with `batch_normalization(training=False)` and `GradientDescentOptimizer`- ```Non-OK-status: status status: Internal: RET_CHECK failure (tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc:32) ShapeUtil::IndexIsValid(alias_.shape(), output_index) Trying to set up alias at {9} which is an invalid index for shape (f32[], f32[2,2,1,1], f32[1], f32[729,256], f32[256], f32[256], f32[256], f32[256,10], f32[10])```
 * cnn + fc layers with `batch_normalization(training=False)` and `GradientDescentOptimizer` - same error as above  
    * In master branch (commit 0886c6e0736135fdc4bbb4905b88158b66955abe), they are adding a read_only option to the variable config, which should solve this issue. Modified function to accept this has been added to the utils.py

Major concerns/Feedback:
 * generating config file manually isn't feasible - script to generate this is in utils.py (for r2.0 and master branch versions (commit 0886c6e0736135fdc4bbb4905b88158b66955abe))
 * naming of variables is extremely constrained, if they can loosen it for the xla generation, and can maintain the constraints for the output file generation that will be ideal.
 * rnn - tf.while_loop isn't using maximum_iterations though its specified error (details in Experiment)

Files:  
* utils.py - contains config generation script and run script to generate graph_def and config script from model_fn and input_fn
* tf2xla_pb2.py - compiled protobuf file for python import
* examples contain the different models.
