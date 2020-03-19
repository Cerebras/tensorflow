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
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"

namespace xla {
namespace {

class HloFileTest : public HloTestBase {
 protected:
  HloFileTest()
      : HloTestBase(
            /*test_platform=*/PlatformUtil::GetPlatform("cpu").ValueOrDie(),
            /*reference_platform=*/PlatformUtil::GetPlatform("cpu")
                .ValueOrDie()) {}
};

TEST_F(HloFileTest, FromCS1Proto) {
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
  auto result = hlo_verifier_->Run(module.get()).status();
  EXPECT_TRUE(result.ok()) << "Verification failed";

  auto fake_arguments = MakeFakeArguments(module.get()).ConsumeValueOrDie();
  std::vector<Literal*> fake_argument_ptrs;
  absl::c_transform(
      fake_arguments, std::back_inserter(fake_argument_ptrs),
      [](const Literal& literal) { return const_cast<Literal*>(&literal); });

  HloRunner hlo_runner(PlatformUtil::GetPlatform("cpu").ValueOrDie());

  auto output_or_status = hlo_runner.Execute(
    std::move(module), fake_argument_ptrs, true /*run_hlo_passes*/);

  std::cout << "OUTPUT: " << output_or_status.ValueOrDie().ToString() << std::endl;
}

}  // namespace
}  // namespace xla
