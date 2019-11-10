#include "tensorflow/tools/xla_extract/tf_graph_to_xla_lib.h"
#include <google/protobuf/util/json_util.h>
#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include <iterator>
#include <string>
#include <tuple>
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"  // for DEVICE_CPU_XLA_JIT
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/graph_execution_state.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/interpreter/compiler.h"

#include <utility>
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/despecializer.h"
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

constexpr const char* PLACEHOLDER = "Placeholder";
constexpr const char* VAR_HANDLE_OP = "VarHandleOp";

static bool verbose = true;

template <typename MSG>
bool save_msg(const MSG& msg, const std::string& file) {
  std::string json;
  google::protobuf::util::JsonPrintOptions op;
  op.add_whitespace = true;
  google::protobuf::util::MessageToJsonString(msg, &json, op);

  FILE* f = fopen(file.c_str(), "wt");
  if (f) {
    fwrite(json.c_str(), json.size(), sizeof(std::string::value_type), f);
    fclose(f);
    return true;
  } else {
    std::cerr << "Could not open file: " << file
              << ", reason: " << strerror(errno) << std::endl
              << std::flush;
    return false;
  }
}

std::vector<XlaCompiler::Argument> BuildXlaArgsFromClientGraph(
    const std::unique_ptr<ClientGraph>& cg) {
  std::vector<XlaCompiler::Argument> xla_args;
  for (const Node* node : cg->graph.nodes()) {
    if (verbose) {
        std::cout << "Inspecting node " << node->name() 
                  << " of type: " << node->type_string() 
                  << std::endl;
    }
    if (node->type_string() == "XlaLaunch") {
      // iterate over the inputs to this node for the args
      for (const Node* in : node->in_nodes()) {
        auto in_def = in->def();
        XlaCompiler::Argument arg;
        const std::string op_name = in_def.op();
        if (verbose) {
            const std::string node_name = in_def.name();
            std::cout << "Node: " << node_name 
                    << ", Op: " << op_name 
                    //<< ", Type: " << in_def.type_string()
                    << std::endl << std::flush;
        }
        if (op_name == "VarHandleOp") {
          arg.kind = XlaCompiler::Argument::kResource;
          arg.resource_kind = XlaResource::kVariable;
          arg.initialized = true;
          Status status = GetNodeAttr(in_def, "shape", &(arg.shape));
          if (!status.ok()) {
            std::cerr << status.error_message() << ", code = " << status.code()
                      << std::endl;
          }
        } else {
          arg.kind = XlaCompiler::Argument::kParameter;
          std::vector<tensorflow::TensorShape> shape_value;
          Status status = GetNodeAttr(in_def, "_output_shapes", &shape_value);
          if (!status.ok()) {
            std::cerr << status.error_message() << ", code = " << status.code() << std::endl;
          }
          std::cout << "shape_value.size() = " << shape_value.size() << " ("
                    << status.error_message() << ")" << std::endl;
          if (status.ok()) {
              assert(!shape_value.empty());
              arg.shape = shape_value[0];
          } else {
            status = GetNodeAttr(in_def, "shape", &(arg.shape));
            if (!status.ok()) {
              std::cerr << status.error_message()
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

void InitializeDevices(const SessionOptions& options, DeviceMgr** device_mgr,
                       DeviceSet* dev_set) {
  std::vector<std::unique_ptr<Device>> devices;
  Status s = DeviceFactory::AddDevices(
      options, "/job:localhost/replica:0/task:0", &devices);
  *device_mgr = new DeviceMgr(std::move(devices));
  //bool have_device = false;
  int devices_added = 0;
  for (auto d : (*device_mgr)->ListDevices()) {
    std::cout << "Found Device: " << d->name() << std::endl << std::flush;
    dev_set->AddDevice(d);
    d->op_segment()->AddHold("HOLD");
    if (devices_added == 0) {
    //if (!have_device) {
      //if (contains(d->name(), DEVICE_CPU_XLA_JIT) || contains(d->name(), DEVICE_XLA_CPU)) {
        std::cout << "Setting client device to: " << d->name() << std::endl << std::flush;
        dev_set->set_client_device(d);
        //have_device = true;
      //}
    //}
    ++devices_added;
   }
  }
  //if (!have_device) {
    //assert(false);
  //}
}

xla::HloModuleProto ExtractHloFromGraphDef(const GraphDef& in_graph,
                                           const std::string& fetch) {
  Status s;
  SessionOptions sess_options;
  DeviceMgr* device_mgr;
  DeviceSet dev_set;
  InitializeDevices(sess_options, &device_mgr, &dev_set);

  // Local copy for modification
  GraphDef gdef = in_graph;
  GraphExecutionStateOptions ges_options;
  ges_options.device_set = &dev_set;
  ges_options.session_options = &sess_options;
  std::unique_ptr<GraphExecutionState> execution_state;
  s = GraphExecutionState::MakeForBaseGraph(&gdef, ges_options,
                                            &execution_state);
  if (!s.ok()) {
    LOG(FATAL) << "execution state creation failed: " << s.error_message();
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
      LOG(FATAL) << "build graph failed " << s.error_message();
  }

  // Usually there is only one cluster, but for some graphs (e.g. LSTM) there
  // may be more.  Return the *last* cluster whose name starts with "cluster_"
  FunctionDefLibrary fdef_lib = client_graph->flib_def->ToProto();

  save_msg(fdef_lib , "/tmp/FunctionDefLibrary.json");

  auto fdef_iter =
      std::find_if(fdef_lib.function().rbegin(), fdef_lib.function().rend(),
                   [](const FunctionDef& f_) -> bool {
                     return (f_.signature().name().find("cluster_") == 0 &&
                             f_.signature().name().substr(
                                 f_.signature().name().length() - 2) == "_0");
                   });

  FunctionDef fdef;

  if (fdef_iter == fdef_lib.function().rend()) {
    fdef_iter =
        std::find_if(fdef_lib.function().rbegin(), fdef_lib.function().rend(),
                     [](const FunctionDef& f_) -> bool {
                       return (f_.signature().name().find("cluster_") == 0);
                     });
  }

  if (fdef_iter != fdef_lib.function().rend()) {
    fdef = *fdef_iter;
  } else {
    fdef = *fdef_lib.function().begin();
    LOG(INFO) << "cluster not found, using " << fdef.signature().name()
              << " instead\n";
  }

  save_msg(fdef, "/tmp/fdef.json");

  auto xla_args = BuildXlaArgsFromClientGraph(client_graph);

  // to make sure xla_args matches fdef

  LOG(INFO) << "number of function defs:" << fdef_lib.function().size() << "\n";
  LOG(INFO) << fdef.signature().name() << "\n";
  LOG(INFO) << "xla args number:" << xla_args.size() << "\n";
  LOG(INFO) << "fdef_args number:" << fdef.signature().input_arg().size()
            << "\n";

  // compares fdef_args(ground truth) with xla_args
  // prunes away extra args and reorders to match fdef_args
  // cant use fdef args directly due to name mismatch
  // we can convert xla_args names to fdef_args names but not vice versa
  auto fdef_ground_truth = fdef.signature().input_arg();
  std::vector<XlaCompiler::Argument> new_xla_args(fdef_ground_truth.size());
  const std::string kReadVarOpString = "readvariableop";
  const std::string kIdentityString = "identity";
  for (int i = 0; i < xla_args.size(); i++) {
    std::string xla_arg_name = xla_args[i].name;
    xla_arg_name = str_util::Lowercase(xla_arg_name);
    xla_arg_name = str_util::ArgDefCase(xla_arg_name);
    xla_arg_name = xla_arg_name + "_0_arg";
    xla_arg_name = str_util::StringReplace(xla_arg_name, kReadVarOpString,
                                           kIdentityString, true);
    for (int j = 0; j < fdef_ground_truth.size(); j++) {
      if (xla_arg_name == fdef_ground_truth[j].name()) {
        new_xla_args[j] = xla_args[i];
      }
    }
  }

  for (int l = 0; l < fdef_ground_truth.size(); l++) {
    if (new_xla_args[l].name == "") {
      LOG(FATAL) << "name mismatch error for " << fdef_ground_truth[l].name();
    }
  }

  xla_args = new_xla_args;
  // we no longer need to do the rotation

  LOG(INFO) << "xla args in correct order and matches fdef\n";
  xla::HloModuleProto hmod;
  {
    DeviceType device_type(DEVICE_CPU_XLA_JIT);
    XlaCompiler::Options compile_options;
    compile_options.client = xla::ClientLibrary::LocalClientOrDie();
    compile_options.device_type = device_type;
    compile_options.flib_def = client_graph->flib_def.get();

    NameAttrList function;
    function.set_name(fdef.signature().name());
    *(function.mutable_attr()) = fdef.attr();

    XlaCompiler compiler(compile_options);
    XlaCompiler::CompilationResult result;

    s = compiler.CompileFunction(XlaCompiler::CompileOptions(), function,
                                 xla_args, &result);
    if (!s.ok()) LOG(FATAL) << "Couldn't compile to xla: " << s.error_message();

    LOG(INFO) << "Done Compiling";
    hmod.CopyFrom(result.computation->proto());

    // hlo optimizations
    xla::StatusOr<xla::ProgramShape> program_shape_status =
        result.computation->GetProgramShape();
    xla::ProgramShape program_shape = program_shape_status.ValueOrDie();
    xla::HloModuleConfig module_config = xla::HloModuleConfig(program_shape);

    xla::StatusOr<std::unique_ptr<xla::HloModule>> hlo_module_status =
        xla::HloModule::CreateFromProto(hmod, module_config);
    std::unique_ptr<xla::HloModule> hlo_module =
        std::move(hlo_module_status.ValueOrDie());

    xla::HloPassPipeline pipeline("Interpreter");

    // adding passes we wish to run
    pipeline.AddPass<xla::CallInliner>();
    pipeline.AddPass<xla::HloSubcomputationUnification>();
    pipeline.AddPass<xla::HloCSE>(false);

    xla::AlgebraicSimplifierOptions options(
        [](const xla::Shape&, const xla::Shape&) { return false; });
    options.set_enable_dot_strength_reduction(false);
    options.set_enable_conv_simplification(false);
    pipeline.AddPass<xla::AlgebraicSimplifier>(options);
    pipeline.AddPass<xla::WhileLoopSimplifier>();
    pipeline.AddPass<xla::ReshapeMover>();
    pipeline.AddPass<xla::HloConstantFolding>();
    pipeline.AddPass<xla::HloCSE>(true);
    pipeline.AddPass<xla::LayoutAssignment>(
        hlo_module.get()->mutable_entry_computation_layout(),
        xla::LayoutAssignment::InstructionCanChangeLayout);
    pipeline.AddPass<xla::HloDCE>();
    pipeline.AddPass<xla::FlattenCallGraph>();

    // hlo optimization run
    s = pipeline.Run(hlo_module.get()).status();

    if (!s.ok())
      LOG(FATAL) << "Couldn't Run HloOptimization: " << s.error_message();

    LOG(INFO) << "Done HLO Optimization\n";
    hmod = hlo_module.get()->ToProto();

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

  if (device_mgr != nullptr) {
    delete (device_mgr);
  }

  return std::move(hmod);
}

Status xla_extract_via_strings(const std::string& graph_def_msg,
                                const std::string& target_node,
                                std::string* out_graph) {
  GraphDef gdef;
  //std::cout << std::endl << graph_def_msg << std::endl;

  gdef.ParseFromString(graph_def_msg);

  save_msg(gdef, "/tmp/graph.json");

  auto hmod = ExtractHloFromGraphDef(gdef, target_node);
  hmod.SerializeToString(out_graph);

  return Status::OK();
}

}  // namespace tensorflow
