/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This demonstrates how to use hlo_test_base to create a file based testcase
// and compare results on gpu and cpu.

#include <string>
#include <vector>
#include <iostream>
#include <fcntl.h>
#include <fstream>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"

#include "tensorflow/compiler/xla/tests/local_client_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

class XLAComputationExecutionTest : public LocalClientTestBase {
 protected:
  ErrorSpec error_spec_{0.0001};
};

XLA_TEST_F(XLAComputationExecutionTest, CompileExecutable) {

  const string& filename = "program_shape_0.pbtxt";
  string test_srcdir = tensorflow::testing::TensorFlowSrcRoot();
  std::cout << "Test dir: " << test_srcdir << std::endl;

  int fileDescriptor = open(tensorflow::io::JoinPath("./", filename).c_str(), O_RDONLY);
  google::protobuf::io::FileInputStream prog_shape_file(fileDescriptor);

  //
  // Steps
  // 0. Read the ProgramShape proto
  ProgramShapeProto prog_shape_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::Parse(&prog_shape_file, &prog_shape_proto));
  close(fileDescriptor);

  // 1. Call mod_cfg = CreateModuleConfigFromProto(cs1-proto)
  ProgramShape prog_shape(prog_shape_proto);
  HloModuleConfig mod_cfg(prog_shape);

  // 2. Call hld_module = CreateFromProto(hlo_mod_proto, mod_cfg)
  fileDescriptor = open(tensorflow::io::JoinPath("./", "hlo_module_0.pbtxt").c_str(), O_RDONLY);
  google::protobuf::io::FileInputStream hmod_proto_file(fileDescriptor);

  HloModuleProto hmod_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::Parse(&hmod_proto_file, &hmod_proto));
  close(fileDescriptor);

  //std::cout << "\n===== HLOModule ======\n" << hmod_proto.DebugString() << std::endl;

  auto hmod_or_status = HloModule::CreateFromProto(hmod_proto, mod_cfg);
  EXPECT_TRUE(hmod_or_status.ok()) << "HLO Module creation failed";

  auto module = hmod_or_status.ConsumeValueOrDie();
  XlaComputation computation(module.get()->ToProto());

  //--------------------------------------------
  const auto params = module->entry_computation()->parameter_instructions();
  std::vector<const Shape*> argument_layouts(params.size());
  for (int i = 0; i < params.size(); ++i) {
    std::cout << "Param: " << params[i]->ToString() << std::endl;
    std::cout << "Category: " << params[i]->ToCategory() << std::endl;
    argument_layouts[i] = &params[i]->shape();
  }

  const Shape& output_shape = module.get()->result_shape();
  std::cout << "RESULT Shape: " <<  output_shape << std::endl;

  auto executable_status =
      local_client_->Compile(computation, argument_layouts,
                             ExecutableBuildOptions());

  std::vector<StatusOr<ScopedShapedBuffer>> fake_input_data_buf(params.size());
  std::vector<ShapedBuffer*> input_data(params.size());
  for (int i = 0; i < params.size(); ++i) {
    auto fake_input_literal = MakeFakeLiteral(params[i]->shape()).ConsumeValueOrDie();
    fake_input_data_buf[i] =
      local_client_->LiteralToShapedBuffer(
        fake_input_literal,
        local_client_->default_device_ordinal());
    input_data[i] = &fake_input_data_buf[i].ValueOrDie();
  }

  ASSERT_IS_OK(executable_status);
  std::unique_ptr<LocalExecutable> executable =
      executable_status.ConsumeValueOrDie();

  ScopedShapedBuffer result =
      executable->Run(input_data, DefaultExecutableRunOptions())
          .ConsumeValueOrDie();
  ASSERT_IS_OK(local_client_->mutable_backend()
                   ->BorrowStream(0)
                   .ValueOrDie()
                   ->BlockHostUntilDone());

  std::cout << "Computation RESULT " << result.ToString() << std::endl;

  // LiteralTestUtil::ExpectR1Near<float>(
  //     {2.0f, 4.0f, 6.0f}, ShapedBufferToLiteral(result), error_spec_);
}

}  // namespace
}  // namespace xla
