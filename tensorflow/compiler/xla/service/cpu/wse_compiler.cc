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

#include "tensorflow/compiler/xla/service/cpu/wse_compiler.h"

#include <stddef.h>
#include <string.h>

#include <map>
#include <mutex>  // NOLINT(build/c++11): only using std::call_once, not mutex.
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <memory>

#include "tensorflow/stream_executor/wse/wse_platform_id.h"
#include "tensorflow/core/util/util.h"

// IWYU pragma: no_include "llvm/Config/Disassemblers.def.inc"
// IWYU pragma: no_include "llvm/Config/Targets.def.inc"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/batch_dot_simplification.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/buffer_liveness.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/cholesky_expander.h"
#include "tensorflow/compiler/xla/service/conditional_simplifier.h"
#include "tensorflow/compiler/xla/service/conditional_to_select.h"
#include "tensorflow/compiler/xla/service/convolution_group_converter.h"
#include "tensorflow/compiler/xla/service/cpu/buffer_info_util.h"
#include "tensorflow/compiler/xla/service/cpu/compiler_functor.h"
#include "tensorflow/compiler/xla/service/cpu/conv_canonicalization.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_copy_insertion.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_hlo_support_checker.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_layout_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_options.h"
#include "tensorflow/compiler/xla/service/cpu/disassembler.h"
#include "tensorflow/compiler/xla/service/cpu/dot_op_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/parallel_task_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"
#include "tensorflow/compiler/xla/service/cpu/wse_compiler.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_element_type_converter.h"
#include "tensorflow/compiler/xla/service/hlo_get_dimension_size_rewriter.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/indexed_array_analysis.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/map_inliner.h"
#include "tensorflow/compiler/xla/service/reduce_precision_insertion.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/scatter_expander.h"
#include "tensorflow/compiler/xla/service/slice_sinker.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/service/triangular_solve_expander.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/while_loop_invariant_code_motion.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/dynamic_annotations.h"

namespace tensorflow {
xla::StatusOr<std::unique_ptr<xla::HloModule>> RunHlo(std::unique_ptr<xla::HloModule>& hlo_module);
}

namespace xla {

namespace wse {

namespace {

template <typename MSG>
std::string msg_to_json(const MSG& msg) {
  std::string json;
  google::protobuf::util::JsonPrintOptions op;
  op.add_whitespace = true;
  google::protobuf::util::MessageToJsonString(msg, &json, op);
  return std::move(json);
}

static std::atomic<int> save_msg_counter{0};

template <typename MSG>
bool save_msg(const MSG& msg, const std::string& file, int counter) {
  const std::string json = msg_to_json(msg);
  const std::string counter_str = std::to_string(counter);
  const std::string json_file = file + "_" + counter_str + ".json";

  FILE* f = fopen(json_file.c_str(), "wt");
  if (f) {
    fwrite(json.c_str(), json.size(), sizeof(std::string::value_type), f);
    fclose(f);
    const std::string pbtxt_file = file + "_" + counter_str + ".pbtxt";
    f = fopen(pbtxt_file.c_str(), "wb");
    if (f) {
      std::string pbtxt;
      msg.SerializeToString(&pbtxt);
      fwrite(pbtxt.data(), pbtxt.size(), sizeof(std::string::value_type), f);
      fclose(f);
      //tensorflow::WriteBinaryProto(tensorflow::Env::Default(), pbtxt_file, msg);
      //tensorflow::WriteTextProto(tensorflow::Env::Default(), pbtxt_file, msg);
      return true;
    }
  } else {
    VLOG(0) << "Could not open file: " << file
            << ", reason: " << strerror(errno) << std::endl
            << std::flush;
    return false;
  }
}

void dump_inputs_outputs(const HloModule& hmod) {
  const HloComputation *entry_comp = hmod.entry_computation();
  std::cout << "******************" << ENDL;
  for (tensorflow::int64 i = 0, n = entry_comp->num_parameters(); i < n; ++i) {
    const HloInstruction *instr = entry_comp->parameter_instruction(i);
    std::cout << "WseCompiler: Input param " << instr->unique_id() << " -> "
              << instr->name()
              << ", shape=" << instr->shape()
              << ENDL;
  }
  std::cout << "==================" << ENDL;
  const HloInstruction* root_instruction = entry_comp->root_instruction();
  for (tensorflow::int64 i = 0, n = root_instruction->operand_count(); i < n; ++i) {
    const HloInstruction *instr = root_instruction->operand(i);
    std::cout << "WseCompiler: Output " << instr->unique_id() << " -> "
              << instr->name()
              << ", shape=" << instr->shape()
              << ENDL;
  }
  std::cout << "******************" << ENDL;
}

}  // namespace (anonymous)

WseCompiler::WseCompiler() {
  // Initialize LLVM the first time the WseCompiler is initialized.
  static bool llvm_initialized = []() {
      InitializeLLVMTarget();
      return true;
  }();
  (void)llvm_initialized;
}

/* static */ void WseCompiler::InitializeLLVMTarget() {
  // Initialize LLVM's MC layer for the native target.
//  llvm::InitializeNativeTarget();
//  llvm::InitializeNativeTargetAsmPrinter();
//  LLVMInitializeX86Target();
//  LLVMInitializeX86TargetInfo();
//  LLVMInitializeX86TargetMC();
//  LLVMInitializeX86AsmPrinter();
//  LLVMInitializeX86Disassembler();
//  LLVMInitializeARMTarget();
//  LLVMInitializeARMTargetInfo();
//  LLVMInitializeARMTargetMC();
//  LLVMInitializeARMAsmPrinter();
//  LLVMInitializeARMDisassembler();
//  LLVMInitializeAArch64Target();
//  LLVMInitializeAArch64TargetInfo();
//  LLVMInitializeAArch64TargetMC();
//  LLVMInitializeAArch64AsmPrinter();
//  LLVMInitializeAArch64Disassembler();
}

std::unique_ptr<HloModule> WseCompiler::copy(const HloModule& src) {
  auto module_proto = src.ToProto();
  DebugOptions debug_options;
  StatusOr<HloModuleConfig> module_config = HloModule::CreateModuleConfigFromProto(
      module_proto, debug_options);
  StatusOr<std::unique_ptr<HloModule>> new_module = HloModule::CreateFromProto(module_proto,
                                                                               module_config.ValueOrDie());
  return std::unique_ptr<HloModule>(new_module.ValueOrDie().release());
}

//std::unique_ptr<HloModuleGroup> WseCompiler::copy(const HloModuleGroup& src) {
//  auto dest_module_group_ptr = absl::make_unique<HloModuleGroup>(src.name());
//  for (size_t i = 0; i < src.size(); ++i) {
//    const HloModule& src_module = src.module(i);
//    dest_module_group_ptr->push_back(copy(src_module));
//  }
//  return std::move(dest_module_group_ptr);
//}

se::Platform::Id WseCompiler::PlatformId() const {
  //return se::host::kHostPlatformId;
  return se::wse::kWsePlatformId;
}

bool WseCompiler::IsEnabled() const {
  if (!this) {
    return false;
  }
  const char *s = getenv("WSE_COMPILER_ENABLED");
  return s && atoi(s) > 0;
}

/**
 * Run Cerebras version of HLO optimizations on the HloModule object
 * @param module
 * @return
 */
StatusOr<std::unique_ptr<HloModule>> WseCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module,
    se::StreamExecutor* /*stream_exec*/,
    se::DeviceMemoryAllocator* /*device_allocator*/) {
  HERE();
  save_msg(module->ToProto(), "wse_hlom_in", save_msg_counter);
  dump_inputs_outputs(*module);
  return tensorflow::RunHlo(module);
}

/**
 * Here is where we'll run the full Cerebras compile
 * @param module
 * @param stream_exec
 * @param device_allocator
 * @return
 */
StatusOr<std::unique_ptr<Executable>> WseCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    se::DeviceMemoryAllocator* device_allocator) {
  //HERE();
  save_msg(module->ToProto(), "wse_hlom_out", save_msg_counter);
  ++save_msg_counter;
  return Status(tensorflow::error::UNIMPLEMENTED, "Cerebras WSE RunBackend() not yet implemented");
}


}  // namespace wse
}  // namespace xla

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      stream_executor::wse::kWsePlatformId,
      []() { return absl::make_unique<xla::wse::WseCompiler>(); });
  return true;
}
static bool module_initialized = InitModule();
