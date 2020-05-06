#include "tensorflow/tools/xla_extract/wse_hlo.h"
#include "tensorflow/tools/xla_extract/utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/layout_assignment.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

#include <google/protobuf/util/json_util.h>

#include <stdexcept>
#include <string>

namespace tensorflow {
namespace wse {

const bool save_messages = get_env_bool("XLA_SAVE_MESSAGES", false);
const bool verbose = get_env_int("XLA_LOG", NO_LOG) >= DEBUG_LOG;

xla::StatusOr<std::unique_ptr<xla::HloModule>> RunHlo(std::unique_ptr<xla::HloModule>& hlo_module) {
  Status s;
  if (verbose) {
    LOG(INFO) << "xla args in correct order and matches fdef\n";
  }
  {
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


    if (get_env_int("XLA_LOG", NO_LOG) >= DEBUG_LOG) {
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
    if (!disable_CallInliner){
      pipeline.AddPass<xla::CallInliner>();
    }
    if (!disable_HloSubcomputationUnification){
      pipeline.AddPass<xla::HloSubcomputationUnification>();
    }
    if (!disable_HloCSE_false){
      pipeline.AddPass<xla::HloCSE>(false);
    }
    if (!disable_AlgebraicSimplifier){
      xla::AlgebraicSimplifierOptions options(
          [](const xla::Shape&, const xla::Shape&) { return false; });
      options.set_enable_dot_strength_reduction(false);
      options.set_enable_conv_simplification(false);
      pipeline.AddPass<xla::AlgebraicSimplifier>(options);
    }
    if (!disable_WhileLoopSimplifier){
      pipeline.AddPass<xla::WhileLoopSimplifier>();
    }
    if (!disable_ReshapeMover){
      pipeline.AddPass<xla::ReshapeMover>();
    }
    if (!disable_HloConstantFolding){
      pipeline.AddPass<xla::HloConstantFolding>();
    }
    if (!disable_HloCSE_true){
      pipeline.AddPass<xla::HloCSE>(true);
    }
    if (!disable_HloDCE){
      pipeline.AddPass<xla::HloDCE>();
    }
    if (!disable_FlattenCallGraph){
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
      return s;
    }

    if (verbose) {
      LOG(INFO) << "Done HLO Optimization\n";
    }

    if (save_messages) {
      std::string out_graph;
      hlo_module->ToProto().SerializeToString(&out_graph);

      if (save_messages) {
        FILE *f = fopen("wse_hlo_xla.pbtxt", "wb");
        assert(f);
        fwrite(out_graph.data(), out_graph.size(), 1, f);
        fclose(f);
        std::string json;
        google::protobuf::util::JsonPrintOptions op;
        op.add_whitespace = true;
        google::protobuf::util::MessageToJsonString(hlo_module->ToProto(),&json, op);
        save_msg(hlo_module->ToProto(), "wse_hlo_xla.json");
      }
    }

  }

  return std::move(hlo_module);
}

}  // namespace wse
}  // namespace tensorflow

