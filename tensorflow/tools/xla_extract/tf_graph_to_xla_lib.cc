#include "tensorflow/tools/xla_extract/tf_graph_to_xla_lib.h"
#include "tensorflow/tools/xla_extract/wse_hlo.h"
#include "tensorflow/tools/xla_extract/utils.h"
#include <google/protobuf/util/json_util.h>
#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"  // for DEVICE_CPU_XLA_JIT
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/graph_execution_state.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/interpreter/compiler.h"

#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/despecializer.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/layout_assignment.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"

#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

namespace {  // anonymous namepace

using namespace wse;

std::vector<XlaCompiler::Argument> BuildXlaArgsFromClientGraph(
    const std::unique_ptr<ClientGraph>& cg) {
  std::vector<XlaCompiler::Argument> xla_args;
  for (const Node* node : cg->graph.nodes()) {
    if (verbose) {
        LOG(INFO) << "Inspecting node " << node->name()
                  << " of type: " << node->type_string()
                  << std::endl;
    }
    if (node->type_string() == "XlaLaunch") {
      // iterate over the inputs to this node for the args
      for (const Node* in : node->in_nodes()) {
        const NodeDef& in_def = in->def();
        XlaCompiler::Argument arg;
        const std::string op_name = in_def.op();
        if (verbose) {
            const std::string node_name = in_def.name();
            LOG(INFO) << "Node: " << node_name << ", Op: " << op_name
                      << ", type: " << in->type_string()
                      << ", req device: " << in->requested_device() << std::endl
                      << std::flush;
        }
        if (op_name == "VarHandleOp") {
          arg.kind = XlaCompiler::Argument::kResource;
          arg.resource_kind = XlaResource::kVariable;
          arg.initialized = true;
          tensorflow::TensorShape shape_value;
          Status status = GetNodeAttr(in_def, "shape", &shape_value);
          arg.shape = shape_value;
          if (!status.ok()) {
            LOG(WARNING) << status.error_message() << ", code = " << status.code()
                         << std::endl;
          }
        } else {
          if (verbose) {
            const std::string node_json = msg_to_json(in_def);
            printf("\n%s\n", node_json.c_str());  fflush(stdout);
          }

          arg.kind = XlaCompiler::Argument::kParameter;
          std::vector<tensorflow::TensorShape> shape_value;
          Status status = GetNodeAttr(in_def, "_output_shapes", &shape_value);
          if (!status.ok()) {
            LOG(WARNING) << status.error_message()
                        << ", code = " << status.code() << std::endl;
          }
          if (verbose) {
            LOG(INFO) << "_output_shapes: shape_value.size() = "
                        << shape_value.size() << " (" << status.error_message()
                        << ")" << std::endl;
          }
          if (status.ok()) {
              assert(!shape_value.empty());
              arg.shape = shape_value[0];
          } else {
            // fall back to 'shape' if there was no '_output_shapes'
            tensorflow::TensorShape shape_value;
            status = GetNodeAttr(in_def, "shape", &shape_value);
            arg.shape = shape_value;
            if (!status.ok()) {
              LOG(ERROR) << status.error_message()
                         << ", code = " << status.code() << std::endl;
            }
          }
        }
        arg.name = in_def.name();

        GetNodeAttr(in_def, "dtype", &(arg.type));
        if (arg.type == DT_INVALID) {
          arg.type = DT_FLOAT;
        }
        xla_args.push_back(std::move(arg));
      }
    }
  }
  return std::move(xla_args);
}

void InitializeDevices(const SessionOptions& options, std::unique_ptr<StaticDeviceMgr>& device_mgr,
                              DeviceSet* dev_set) {
  std::vector<std::unique_ptr<Device>> devices;
  Status s = DeviceFactory::AddDevices(options, "/job:localhost/replica:0/task:0", &devices);
  device_mgr = std::make_unique<StaticDeviceMgr>(std::move(devices));
  bool have_device = false;
  int devices_added = 0;
  for (Device *d : device_mgr->ListDevices()) {
    const std::string device_type = d->device_type();
    if (verbose) {
      LOG(INFO) << "Found Device: " << d->name() << " (" << device_type << ")"
                << std::endl
                << std::flush;
    }
    // GPU devices alter the client graph in an incompatible way to the curent implementation
    //if (device_type == DEVICE_XLA_CPU || device_type == "CPU" || device_type == DEVICE_CPU_XLA_JIT) {
    if (device_type == DEVICE_XLA_CPU || device_type == "CPU") {
      dev_set->AddDevice(d);
      d->op_segment()->AddHold("HOLD");
      const std::string& device_name = d->name();
      if (!have_device) {
        if (device_type == "CPU") {
          if (verbose) {
            LOG(INFO) << "Setting client device to: " << device_name << std::endl
                      << std::flush;
          }
          dev_set->set_client_device(d);
          have_device = true;
        }
        ++devices_added;
      }
    }
  }
  if (!have_device) {
      throw std::runtime_error("Did not find a suitable client device");
  }
  if (verbose) {
    LOG(INFO) << "Added " << devices_added << " devices" << std::endl << std::flush;
  }
}

/**
 * @brief Get the Compile Platform object (i.e. "Host", "CUDA", etc.)
 *
 * @return se::Platform* Pointer to Platform object to use for compile (prefer "Host")
 */
se::Platform* getCompilePlatform() {
    xla::StatusOr<std::vector<se::Platform*>> sop = xla::PlatformUtil::GetSupportedPlatforms();
    std::vector<se::Platform*>& platforms = sop.ValueOrDie();
    se::Platform *platform = nullptr;
    for (se::Platform *p : platforms) {
        if (verbose) {
            LOG(INFO) << "Found platform: " << p->Name() << std::endl;
        }
        // Get first one, or take "Host" if found
        if (!platform || p->Name() == "Host") {
            platform = p;
        }
    }
    return platform;
}

}  // anonymous namespace

xla::HloModuleProto ExtractHloFromGraphDef(GraphDef&& in_graph,
                                           const std::string& fetch) {
  Status s;
  SessionOptions sess_options;
  sess_options.config.mutable_graph_options()->mutable_rewrite_options()->set_memory_optimization(RewriterConfig::NO_MEM_OPT);
  std::unique_ptr<StaticDeviceMgr> device_mgr;
  DeviceSet dev_set;
  // XLA_LOG == 0, no prints
  // XLA_LOG == 1, final message only
  // XLA_LOG == 2, other useful messages
  InitializeDevices(sess_options, device_mgr, &dev_set);

  const std::string dtype = dev_set.client_device()->device_type();

  // Local copy for modification
  GraphDef _gdef = in_graph;
  GraphExecutionStateOptions ges_options;
  ges_options.device_set = &dev_set;
  ges_options.session_options = &sess_options;
  std::unique_ptr<GraphExecutionState> execution_state;
  s = GraphExecutionState::MakeForBaseGraph(std::move(_gdef), ges_options,
                                            &execution_state);
  if (!s.ok()) {
    LOG(ERROR) << "execution state creation failed: " << s.error_message();
    throw std::runtime_error(s.error_message());
  }
  BuildGraphOptions bg_options;
  bg_options.use_function_convention = true;
  std::istringstream fetch_stream(fetch);
  std::vector<std::string> fetches(
      std::istream_iterator<std::string>{fetch_stream},
      std::istream_iterator<std::string>());
  for (const std::string& fetch : fetches) {
    bg_options.callable_options.add_fetch(fetch);
  }
  std::unique_ptr<ClientGraph> client_graph;
  s = execution_state->BuildGraph(bg_options, &client_graph);
  if (!s.ok()) {
      LOG(ERROR) << "build graph failed " << s.error_message();
  }

  // Usually there is only one cluster, but for some graphs (e.g. LSTM) there
  // may be more.  Return the *last* cluster whose name starts with "cluster_"
  tensorflow::FunctionDefLibrary fdef_lib = client_graph->flib_def->ToProto();

  if (save_messages) {
    save_msg(fdef_lib , "FunctionDefLibrary.json");
  }

  auto fdef_iter =
      std::find_if(fdef_lib.mutable_function()->rbegin(), fdef_lib.mutable_function()->rend(),
                   [](const FunctionDef& f_) -> bool {
                     return (f_.signature().name().find("cluster_") == 0 &&
                             f_.signature().name().substr(
                                 f_.signature().name().length() - 2) == "_0");
                   });

  if (fdef_iter == fdef_lib.mutable_function()->rend()) {
    fdef_iter =
        std::find_if(fdef_lib.mutable_function()->rbegin(), fdef_lib.mutable_function()->rend(),
                     [](const FunctionDef& f_) -> bool {
                       return (f_.signature().name().find("cluster_") == 0);
                     });
  }

  if (fdef_iter == fdef_lib.mutable_function()->rend()) {
    fdef_iter = fdef_lib.mutable_function()->rend()-1;
    FunctionDef temp_fdef = *fdef_iter;

    if (verbose) {
      LOG(INFO) << "cluster not found, using " << temp_fdef.signature().name()
                << " instead\n";
    }
  }

  const FunctionDef& fdef = *fdef_iter;

  if (save_messages) {
    save_msg(fdef, "fdef.json");
  }

  std::vector<XlaCompiler::Argument> xla_args = BuildXlaArgsFromClientGraph(client_graph);

  // to make sure xla_args matches fdef
  if (wse::get_env_int("XLA_LOG", NO_LOG) >= INFO_LOG) {
    LOG(INFO) << "number of function defs:" << fdef_lib.function().size() << std::endl;
    LOG(INFO) << fdef.signature().name() << "\n";
    LOG(INFO) << "xla args number:" << xla_args.size() << std::endl;
    LOG(INFO) << "fdef_args number:" << fdef.signature().input_arg().size()
              << std::endl;
    // fdef.ret:
    // A mapping from the output arg names from `signature` to the
    // outputs from `node_def` that should be returned by the function.
    LOG(INFO) << "fdef output mapping signature -> node_def: " << std::endl;
    for (std::pair<std::string, std::string> signature_to_node_def_output : fdef.ret()) {
      LOG(INFO) << "\t\"" << signature_to_node_def_output.first << "\" -> \""
                << signature_to_node_def_output.second << "\""
                << std::endl;
    }
  }

  std::list<std::string> output_order;  // TODO: Should return this
  const OpDef& signature = fdef.signature();
  assert(signature.output_arg_size() == fdef.ret_size());
  for (size_t i = 0, n = signature.output_arg_size(); i < n; ++i) {
    const ::tensorflow::OpDef_ArgDef& arg = signature.output_arg(i);
    const std::string& output_name = arg.name();
    const auto iter = fdef.ret().find(output_name);
    assert(iter != fdef.ret().end());
    std::string incoming_output_name = iter->second;
    static const std::string output_str = ":output:";
    std::size_t output_pos = incoming_output_name.find_last_of(output_str);
    if (output_pos != std::string::npos) {
      const std::string fixname1 = incoming_output_name.substr(0, output_pos - (output_str.size() - 1));
      const std::string fixname2 = incoming_output_name.substr(output_pos);
      incoming_output_name = fixname1;
      incoming_output_name += fixname2;
    }
    std::cout << "Incoming output name: " << incoming_output_name << std::endl << std::flush;
    output_order.push_back(incoming_output_name);
  }

  // compares fdef_args(ground truth) with xla_args
  // prunes away extra args and reorders to match fdef_args
  // cant use fdef args directly due to name mismatch
  // we can convert xla_args names to fdef_args names but not vice versa
  auto fdef_ground_truth = fdef.signature().input_arg();
  std::vector<XlaCompiler::Argument> new_xla_args(fdef_ground_truth.size());
  const std::string kReadVarOpString = "readvariableop";
  const std::string kIdentityString = "identity";

  // additional case check
  bool match_flag = false;
  bool readvarop_flag = false;
  for (int j = 0; j < fdef_ground_truth.size(); j++) {
    std::size_t found = fdef_ground_truth[j].name().find(kReadVarOpString);
    if (found != std::string::npos){
      readvarop_flag = true;
    }
  }
  //

  for (int i = 0; i < xla_args.size(); i++) {
    match_flag = false;
    std::string xla_arg_name = xla_args[i].name;
    xla_arg_name = str_util::Lowercase(xla_arg_name);
    xla_arg_name = str_util::ArgDefCase(xla_arg_name);
    xla_arg_name = xla_arg_name + "_0_arg";
    xla_arg_name = str_util::StringReplace(xla_arg_name, kReadVarOpString,
                                           kIdentityString, true);
    for (int j = 0; j < fdef_ground_truth.size(); j++) {
      if (xla_arg_name == fdef_ground_truth[j].name()) {
        new_xla_args[j] = xla_args[i];
        match_flag = true;
        break;
      }
    }
    // additional case check
    if (!match_flag && readvarop_flag){
        std::string xla_arg_name = xla_args[i].name;
    xla_arg_name = str_util::Lowercase(xla_arg_name);
    xla_arg_name = str_util::ArgDefCase(xla_arg_name);
    xla_arg_name = xla_arg_name + "_0_arg";
    for (int j = 0; j < fdef_ground_truth.size(); j++) {
      if (xla_arg_name == fdef_ground_truth[j].name()) {
        new_xla_args[j] = xla_args[i];
        break;
      }
    }
    }
    //
  }

  for (int l = 0; l < fdef_ground_truth.size(); l++) {
    if (new_xla_args[l].name == "") {
      LOG(ERROR) << "name mismatch error for " << fdef_ground_truth[l].name();
    }
  }

  xla_args = new_xla_args;
  // we no longer need to do the rotation

  if (verbose) {
    LOG(INFO) << "xla args in correct order and matches fdef\n";
  }
  xla::HloModuleProto hmod;
  {
    DeviceType device_type(DEVICE_CPU_XLA_JIT);
    XlaCompiler::Options compile_options;

    se::Platform *platform = getCompilePlatform();
    if (!platform) {
        throw std::runtime_error("Could not determine platform for compile");
    }
    if (verbose) {
        LOG(INFO) << "Using platform: " << platform->Name() << std::endl << std::flush;
    }
    auto soc = xla::ClientLibrary::GetOrCreateCompileOnlyClient(platform);
    compile_options.client = soc.ValueOrDie();
    compile_options.device_type = device_type;
    compile_options.flib_def = client_graph->flib_def.get();

    NameAttrList function;
    function.set_name(fdef.signature().name());
    *(function.mutable_attr()) = fdef.attr();

    XlaCompiler compiler(compile_options);
    XlaCompiler::CompilationResult result;

    s = compiler.CompileFunction(XlaCompiler::CompileOptions(), function,
                                 xla_args, &result);
    if (!s.ok()) {
      std::string msg = "Couldn't compile to xla: ";
      msg += s.error_message();
      LOG(ERROR) << msg;
      if (!result.computation.get()) {
        throw std::runtime_error(msg);
      }
    }

    if (verbose) {
      LOG(INFO) << "Done Compiling";
    }

    if (result.computation.get()) {
      hmod.CopyFrom(result.computation->proto());
    }

    if (save_messages) {
      save_msg(hmod, "hmod_in.json");
    }

    // hlo optimizations
    xla::StatusOr<xla::ProgramShape> program_shape_status =
        result.computation->GetProgramShape();
    xla::ProgramShape program_shape = program_shape_status.ValueOrDie();
    xla::HloModuleConfig module_config = xla::HloModuleConfig(program_shape);

    xla::StatusOr<std::unique_ptr<xla::HloModule>> hlo_module_status =
        xla::HloModule::CreateFromProto(hmod, module_config);
    std::unique_ptr<xla::HloModule> hlo_module =
        std::move(hlo_module_status.ValueOrDie());

    auto hlo_run_result = RunHlo(hlo_module);
    if (!hlo_run_result.ok()) {
      throw std::runtime_error(hlo_run_result.status().error_message());
    }
    hmod = hlo_run_result.ValueOrDie().get()->ToProto();

    if (save_messages) {
      save_msg(hmod, "hmod_out_1.json");
    }

    auto* comps = hmod.mutable_computations();

    auto entry_comp_iter =
        std::find_if(comps->begin(), comps->end(),
                     [&hmod](const xla::HloComputationProto& c_) -> bool {
                       return c_.id() == hmod.entry_computation_id();
                     });

    if (entry_comp_iter == comps->end()) {
      throw std::runtime_error(
          "Could not find entry computation in HLO module.");
    }
    xla::HloComputationProto& entry_comp = *entry_comp_iter;

    std::for_each(entry_comp.mutable_instructions()->begin(),
                  entry_comp.mutable_instructions()->end(),
                  [&xla_args](xla::HloInstructionProto& instr) {
                    if (instr.opcode() == "parameter") {
                      instr.set_name(xla_args[instr.parameter_number()].name);
                    }
                  });
  }

  if (save_messages) {
    save_msg(hmod, "hmod_out_2.json");
  }

  return std::move(hmod);
}

Status xla_extract_via_strings(const std::string& graph_def_msg,
                               const std::string& target_node,
                               std::string* out_graph) {
  GraphDef gdef;
  gdef.ParseFromString(graph_def_msg);

  if (save_messages) {
    save_msg(gdef, "graph.json");
  }

  auto hmod = ExtractHloFromGraphDef(std::move(gdef), target_node);
  hmod.SerializeToString(out_graph);

  if (save_messages) {
    FILE *f = fopen("xla_module.pbtxt", "wb");
    assert(f);
    fwrite(out_graph->data(), out_graph->size(), 1, f);
    fclose(f);
    save_msg(hmod, "xla_module.json");
  }

  if (get_env_int("XLA_LOG", NO_LOG) >= INFO_LOG) {
      std::cout << "XLA Extraction Complete\n";
  }

  return Status::OK();
}

}  // namespace tensorflow
