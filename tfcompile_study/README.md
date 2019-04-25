## tfcompile
 Exploration of the tfcompile tool to see how to use it, its limitations and if it could be a suitable alternative to our current xla extraction tool.

1. **Tensorflow Version:**
  * Based of tensorflow r2.0 branch (to build we need to update git > 2.0)
  * Branch used can be found at https://github.com/Cerebras/tensorflow/tree/vishal/tfcompile
  *  This tf branch was build using the default settings in `./configure`.

  * Only change in the code base is in `tensorflow/compiler/aot/codegen.cc#L774` to support `/` in node names.  
    * Originally it supports names that follow C++11 Standard naming convention, so this change to handle variable_scope.

2. **Inputs required for tfcompile:**

  * `tfcompile` command line tool requires at least:
    1. `--graph` (.pbtxt or .pb)
      * graph is extracted from the model (setting `add_shapes=True`, did not require `xla.compile`)  
    2. `--config` (.pbtxt or .pb)
      * This contains `feed` (input) nodes, `fetch` (output) nodes and `variable` nodes.
      * The examples in the Tensorflow repository, are all generated manually, but for those examples and the ones we tested, we generated the config file based of the graph
      * the config is based on the proto file (`tensorflow/compiler/tf2xla/tf2xla.proto`)
    3. `--cpp_class` (for generated .h  and .o files)
      * cpp_class is set to the same as in their examples (`mynamespace::MyComputation`)

3. **To compile tf2xla.proto**  
```bash
protoc --python_out=/path_to_store_compiled_file --proto_path=/path_to_tensorflow_dir/tensorflow   tensorflow/compiler/tf2xla/tf2xla.proto
```

4. **To run tfcompile:**
```bash
TF_CPP_MIN_VLOG_LEVEL=3 /path_to_tensorflow/tensorflow/bazel-bin/tensorflow/compiler/aot/tfcompile --graph=graph_model_fn.pbtxt --config=config_model_fn.config.pbtxt --cpp_class="mynamespace::MyComputation"
```

5. **Files to replicate experiments:**  
  1. utils.py - contains config generation script and run script to generate the `graph_def` and the config script from the `model_fn` and the `input_fn`.
  2. tf2xla_pb2.py - compiled protobuf file of `tf2xla.proto` for python import
  3. examples - directory with each subdirectory containing a specific model.

6. **Experiments:**
  1. Experiments which passed: (able to compile successfully and viewed xla through env variable (`TF_CPP_MIN_VLOG_LEVEL=3`))
    1. fully connected layers with `GradientDescentOptimizer`
    2. fully connected layers with `AdamOptimizer`
    3. fully connected layers with `batch_normalization(training=True)` and `GradientDescentOptimizer`
    4. fully connected layers with `batch_normalization(training=True)` and `AdamOptimizer`
    5. fully connected layers with `keras.BatchNormalization()` works if updates are captured and added (based of Min's PR (#17746) for Xception)
    6. cnn + fc layers with `GradientDescentOptimizer`
    7. cnn + fc layers with `AdamOptimizer`
    8. cnn + fc layers with `batch_normalization(training=True)` and `GradientDescentOptimizer`
    9. cnn + fc layers with `batch_normalization(training=True)` and `AdamOptimizer`

  2. Experiments which failed:
    1. `BasicRNNCell` with `dynamic_rnn` - ```INVALID ARGUMENTS: XLA compilation requires a fixed stack size upper bound. If you are using tf.while_loop, set the maximum_iterations parameter to fix this issue.
  	 [[{{node dynamic_rnn/gradients/dynamic_rnn/func/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc}}]]```
      * Both with and without specifying sequence_length failed.
      *  In our extract tool, if sequence_length is None, it can extract xla successfully.
      *  While setting up our extract tool, rnns would generate two function defs (my understanding was one was xla_cpu, and other was xla_cpu_jit specific), the xla_cpu_jit specific version would give a similar error as above (since tfcompile uses xla_cpu_jit as device, I think its the same issue.)
    2. Using `tf.keras.layers.SimpleRNNCell` and `tf.keras.layers.RNN` - same error as above
    3. fully connected layers with `batch_normalization(training=False)` and `GradientDescentOptimizer`- ```Non-OK-status: status status: Internal: RET_CHECK failure (tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc:32) ShapeUtil::IndexIsValid(alias_.shape(), output_index) Trying to set up alias at {9} which is an invalid index for shape (f32[], f32[2,2,1,1], f32[1], f32[729,256], f32[256], f32[256], f32[256], f32[256,10], f32[10])```
    4. cnn + fc layers with `batch_normalization(training=False)` and `GradientDescentOptimizer` - same error as above  
      * In master branch (`commit 0886c6e0736135fdc4bbb4905b88158b66955abe`), they are adding a read_only option to the variable config, which should solve this issue. Modified function to accept this has been added to the utils.py
      * To make it work in 2.0. is to remove the corresponding variables manually, but only way would be to do a name comparison, which isn't ideal.

7. **Concerns and feedback:**
  1. Generating config file manually isn't feasible - script to generate for our examples is provided in utils.py (for r2.0 and master branch versions (`commit 0886c6e0736135fdc4bbb4905b88158b66955abe`))
  2. Naming of variables is extremely constrained, if they can loosen it for the xla generation, and can maintain the constraints for the output file generation that will be ideal.
  3. RNN - tf.while_loop isn't using maximum_iterations though its specified error (details in Experiment)
