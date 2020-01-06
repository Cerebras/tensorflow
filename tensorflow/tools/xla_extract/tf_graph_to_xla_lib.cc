#include "tensorflow/tools/xla_extract/tf_graph_to_xla_lib.h"
#include <google/protobuf/util/json_util.h>
#include <stdio.h>
#include <unistd.h>
#include <algorithm>
#include <iterator>
#include <string>
#include <tuple>
#include <utility>
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

#define NO_LOG 0
#define INFO_LOG 1
#define DEBUG_LOG 2

namespace tensorflow {

namespace {  // anonymous namepace

bool is_true(const char *s) {
    if (s && *s) {
        const char c = ::tolower(*s);
        if (c == 'y' || c == 't') {
            return true;
        }
        return atoi(s) > 0;
    }
    return false;
}

bool get_env_bool(const char *s, const bool dflt) {
    const char *v = getenv(s);
    if (v && *v) {
      return is_true(v);
    }
    return dflt;
}

int get_env_int(const char *s, const int dflt) {
  const char* v = getenv(s);
  if (v && *v) {
    return atoi(v);
  }
  return dflt;
}

const bool save_messages = get_env_bool("XLA_SAVE_MESSAGES", false);
const bool verbose = get_env_int("XLA_LOG", NO_LOG) >= DEBUG_LOG;

template <typename MSG>
std::string msg_to_json(const MSG& msg) {
  std::string json;
  google::protobuf::util::JsonPrintOptions op;
  op.add_whitespace = true;
  google::protobuf::util::MessageToJsonString(msg, &json, op);
  return std::move(json);
}

template <typename MSG>
bool save_msg(const MSG& msg, const std::string& file) {
  const std::string json = msg_to_json(msg);

  FILE* f = fopen(file.c_str(), "wt");
  if (f) {
    fwrite(json.c_str(), json.size(), sizeof(std::string::value_type), f);
    fclose(f);
    return true;
  } else {
    LOG(ERROR) << "Could not open file: " << file
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

void InitializeDevices(const SessionOptions& options, DeviceMgr** device_mgr,
                              DeviceSet* dev_set) {
  std::vector<std::unique_ptr<Device>> devices;
  Status s = DeviceFactory::AddDevices(options, "/job:localhost/replica:0/task:0", &devices);
  *device_mgr = new DeviceMgr(std::move(devices));
  bool have_device = false;
  int devices_added = 0;
  for (Device *d : (*device_mgr)->ListDevices()) {
    const std::string device_type = d->device_type();
    if (verbose) {
      LOG(INFO) << "Found Device: " << d->name() << " (" << device_type << ")"
                << std::endl
                << std::flush;
    }
    // GPU devices alter the client graph in an incompatible way to the curent implementation
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
};

}  // anonymous namespace

xla::HloModuleProto ExtractHloFromGraphDef(const GraphDef& in_graph,
                                           const std::string& fetch) {
  Status s;
  SessionOptions sess_options;
  sess_options.config.mutable_graph_options()->mutable_rewrite_options()->set_memory_optimization(RewriterConfig::NO_MEM_OPT);
  DeviceMgr* device_mgr;
  DeviceSet dev_set;
  // XLA_LOG == 0, no prints
  // XLA_LOG == 1, final message only
  // XLA_LOG == 2, other useful messages
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
    LOG(ERROR) << "execution state creation failed: " << s.error_message();
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
  FunctionDefLibrary fdef_lib = client_graph->flib_def->ToProto();

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

    if(verbose) {
      LOG(INFO) << "cluster not found, using " << temp_fdef.signature().name()
                << " instead\n";
    }
  }

  FunctionDef& fdef = *fdef_iter;

  if (save_messages) {
    save_msg(fdef, "fdef.json");
  }

  std::vector<XlaCompiler::Argument> xla_args = BuildXlaArgsFromClientGraph(client_graph);

  // to make sure xla_args matches fdef
  if(verbose) {
    LOG(INFO) << "number of function defs:" << fdef_lib.function().size() << "\n";
    LOG(INFO) << fdef.signature().name() << "\n";
    LOG(INFO) << "xla args number:" << xla_args.size() << "\n";
    LOG(INFO) << "fdef_args number:" << fdef.signature().input_arg().size()
              << "\n";
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
    if(found != std::string::npos){
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
    if(!match_flag && readvarop_flag){
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
    if (!s.ok()) LOG(ERROR) << "Couldn't compile to xla: " << s.error_message();


    if (verbose) {
      LOG(INFO) << "Done Compiling";
    }
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
    const bool disable_CallInliner = get_env_bool("DISABLE_CALL_INLINER", false);
    const bool disable_HloSubcomputationUnification = get_env_bool("DISABLE_HLO_SUBCOMPUTATION_UNIFICATION", false);
    const bool disable_HloCSE_false = get_env_bool("DISABLE_HLO_CSE_FALSE", false);
    const bool disable_AlgebraicSimplifier = get_env_bool("DISABLE_ALGEBRAIC_SIMPLIFIER", false);
    const bool disable_WhileLoopSimplifier = get_env_bool("DISABLE_WHILE_LOOP_SIMPLIFIER", false);
    const bool disable_ReshapeMover = get_env_bool("DISABLE_RESHAPE_MOVER", false);
    const bool disable_HloConstantFolding = get_env_bool("DISABLE_HLO_CONSTANT_FOLDING", false);
    const bool disable_HloCSE_true = get_env_bool("DISABLE_HLO_CSE_TRUE", false);
    const bool disable_HloDCE = get_env_bool("DISABLE_HLO_DCE", false);
    const bool disable_FlattenCallGraph = get_env_bool("DISABLE_FLATTEN_CALL_GRAPH", false);


    if(get_env_int("XLA_LOG", NO_LOG) >= DEBUG_LOG) {
      std::cout << "DISABLE_CALL_INLINER: "<< disable_CallInliner<<"\n";
      std::cout << "DISABLE_HLO_SUBCOMPUTATION_UNIFICATION: "<< disable_HloSubcomputationUnification<<"\n";
      std::cout << "DISABLE_HLO_CSE_FALSE: "<< disable_HloCSE_false<<"\n";
      std::cout << "DISABLE_ALGEBRAIC_SIMPLIFIER: "<< disable_AlgebraicSimplifier<<"\n";
      std::cout << "DISABLE_WHILE_LOOP_SIMPLIFIER: "<< disable_WhileLoopSimplifier<<"\n";
      std::cout << "DISABLE_RESHAPE_MOVER: "<< disable_ReshapeMover<<"\n";
      std::cout << "DISABLE_HLO_CONSTANT_FOLDING: "<< disable_HloConstantFolding<<"\n";
      std::cout << "DISABLE_HLO_CSE_TRUE: "<< disable_HloCSE_true<<"\n";
      std::cout << "DISABLE_HLO_DCE: "<< disable_HloDCE<<"\n";
      std::cout << "DISABLE_FLATTEN_CALL_GRAPH: "<< disable_FlattenCallGraph<<"\n";
    }
    if(disable_CallInliner==false){
      pipeline.AddPass<xla::CallInliner>();
    }
    if(disable_HloSubcomputationUnification==false){
      pipeline.AddPass<xla::HloSubcomputationUnification>();
    }
    if(disable_HloCSE_false==false){
      pipeline.AddPass<xla::HloCSE>(false);
    }
    if(disable_AlgebraicSimplifier==false){
      xla::AlgebraicSimplifierOptions options(
        [](const xla::Shape&, const xla::Shape&) { return false; });
      options.set_enable_dot_strength_reduction(false);
      options.set_enable_conv_simplification(false);
      pipeline.AddPass<xla::AlgebraicSimplifier>(options);
    }
    if(disable_WhileLoopSimplifier==false){
      pipeline.AddPass<xla::WhileLoopSimplifier>();
    }
    if(disable_ReshapeMover==false){
      pipeline.AddPass<xla::ReshapeMover>();
    }
    if(disable_HloConstantFolding==false){
      pipeline.AddPass<xla::HloConstantFolding>();
    }
    if(disable_HloCSE_true==false){
      pipeline.AddPass<xla::HloCSE>(true);
    }
    if(disable_HloDCE==false){
      pipeline.AddPass<xla::HloDCE>();
    }
    if(disable_FlattenCallGraph==false){
      pipeline.AddPass<xla::FlattenCallGraph>();
    }

    /*disabled since it errors out
    pipeline.AddPass<xla::LayoutAssignment>(
        hlo_module.get()->mutable_entry_computation_layout(),
        xla::LayoutAssignment::InstructionCanChangeLayout);
    */

    // hlo optimization run
    s = pipeline.Run(hlo_module.get()).status();

    if (!s.ok()) {
      LOG(ERROR) << "Couldn't Run HloOptimization: " << s.error_message();
    }

    if (verbose) {
      LOG(INFO) << "Done HLO Optimization\n";
    }
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

  gdef.ParseFromString(graph_def_msg);

  if (save_messages) {
    save_msg(gdef, "graph.json");
  }

  auto hmod = ExtractHloFromGraphDef(gdef, target_node);
  hmod.SerializeToString(out_graph);

  if(get_env_int("XLA_LOG", NO_LOG) >= INFO_LOG) {
      std::cout << "XLA Extraction Complete\n";
  }

  return Status::OK();
}

}  // namespace tensorflow
