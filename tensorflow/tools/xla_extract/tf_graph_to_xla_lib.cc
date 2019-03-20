#include <stdio.h>
#include <unistd.h>
#include <iterator>
#include <string>
#include <tuple>
#include <algorithm>
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"  // for DEVICE_CPU_XLA_JIT
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/graph_execution_state.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/tools/xla_extract/tf_graph_to_xla_lib.h"

#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/interpreter/compiler.h"

#include "tensorflow/compiler/xla/service/layout_assignment.h"
#include "tensorflow/compiler/xla/service/despecializer.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include <utility>
namespace tensorflow {

std::vector<XlaCompiler::Argument> BuildXlaArgsFromClientGraph(
    const std::unique_ptr<ClientGraph>& cg) {
  std::vector<XlaCompiler::Argument> xla_args;
  for (const Node* node: cg->graph.nodes()) {
    if (node->type_string() == "XlaLaunch") {
      // iterate over the inputs to this node for the args
      for (const Node* in : node->in_nodes()) {
        auto in_def = in->def();
        XlaCompiler::Argument arg;
        if (in_def.op() == "VarHandleOp") {
          arg.kind = XlaCompiler::Argument::kResource;
          arg.resource_kind = XlaResource::kVariable;
          arg.initialized = true;
          GetNodeAttr(in_def, "shape", &(arg.shape));
        } else {
          arg.kind = XlaCompiler::Argument::kParameter;
          std::vector<tensorflow::TensorShape> shape_value;
          GetNodeAttr(in_def, "_output_shapes", &shape_value);
          arg.shape = shape_value[0];
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
  int devices_added = 0;
  for (auto d : (*device_mgr)->ListDevices()) {
    dev_set->AddDevice(d);
    d->op_segment()->AddHold("HOLD");
    if (devices_added == 0) {
      dev_set->set_client_device(d);
    }
    ++devices_added;
  }
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
  if (!s.ok())
    LOG(FATAL) << "execution state creation failed: " << s.error_message();
  BuildGraphOptions bg_options;
  bg_options.use_function_convention = true;
  std::istringstream fetch_stream(fetch);
  std::vector<std::string> fetches(
      std::istream_iterator<std::string>{fetch_stream},
      std::istream_iterator<std::string>());
  for (std::string fetch0 : fetches) {
    bg_options.callable_options.add_fetch(fetch0);
  }
  std::unique_ptr<ClientGraph> client_graph;
  s = execution_state->BuildGraph(bg_options, &client_graph);
  if (!s.ok()) LOG(FATAL) << "build graph failed " << s.error_message();

  // Usually there is only one cluster, but for some graphs (e.g. LSTM) there
  // may be more.  Return the *last* cluster whose name starts with "cluster_"
  FunctionDefLibrary fdef_lib = client_graph->flib_def->ToProto();

  // to be removed
  int n = fdef_lib.function().size();
  std::cout<<"number of fdef_lib function "<<n<<"\n";
  for(int i=0;i<n;i++){
    std::cout<<fdef_lib.function()[i].signature().name()<<" \n";
  }
  // end
  auto fdef_iter = std::find_if(fdef_lib.function().rbegin(), fdef_lib.function().rend(),
				[] (const FunctionDef& f_) -> bool {
				  return (f_.signature().name().find("cluster_") == 0 &&
                  f_.signature().name().substr(f_.signature().name().length() - 2)=="_0");
				});

  FunctionDef fdef;

  if(fdef_iter == fdef_lib.function().rend()){
    fdef_iter = std::find_if(fdef_lib.function().rbegin(), fdef_lib.function().rend(),
				[] (const FunctionDef& f_) -> bool {
				  return (f_.signature().name().find("cluster_") == 0);
				});
  }

  if(fdef_iter != fdef_lib.function().rend()){
    fdef = *fdef_iter;
  }
  else{
    fdef = *fdef_lib.function().begin();
    LOG(INFO) << "cluster not found, using "<<fdef.signature().name()<<" instead\n";
  }


  auto xla_args = BuildXlaArgsFromClientGraph(client_graph);


  // to make sure xla_args matches fdef
  std::vector<XlaCompiler::Argument> new_xla_args;
  auto fdef_ground = fdef.signature().input_arg();

  for(int i = 0;i<xla_args.size();i++){
    std::string xla_arg_name=xla_args[i].name;
    std::replace(xla_arg_name.begin(),xla_arg_name.end(), '/','_');
    std::replace(xla_arg_name.begin(),xla_arg_name.end(), '.','_');
    xla_arg_name=xla_arg_name+"_0_arg";
    std::transform(xla_arg_name.begin(), xla_arg_name.end(), xla_arg_name.begin(), ::tolower);

    std:string readvar="readvariableop";
    std::string read_idt="identity";
    std::size_t pos = xla_arg_name.find(readvar);

    if(pos!= std::string::npos){
      xla_arg_name.replace(pos,readvar.length(),read_idt);
    }

    for(int j=0; j<fdef_ground.size();j++){
        if(xla_arg_name==fdef_ground[j].name()){
          new_xla_args.push_back(xla_args[i]);
        }
    }

  }
  xla_args=new_xla_args;

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
      [&xla_args] (xla::HloInstructionProto& instr) {
        if (instr.opcode() == "parameter") {
          instr.set_name(xla_args[instr.parameter_number()].name);
        }
      });
  }

  if (device_mgr != nullptr) {
    delete(device_mgr);
  }

  return std::move(hmod);
}

Status xla_extract_via_strings(const std::string& graph_def_msg,
             const std::string& target_node,
             std::string* out_graph) {
  GraphDef gdef;
  gdef.ParseFromString(graph_def_msg);
  auto hmod = ExtractHloFromGraphDef(gdef, target_node);
  hmod.SerializeToString(out_graph);

  return Status::OK();
}

}  // namespace tensorflow
