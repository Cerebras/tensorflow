#include <stdio.h>
#include <unistd.h>
#include <string>

#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/tools/xla_extract/tf_graph_to_xla_lib.h"

namespace tensorflow {

void RealMain(const std::string& in_graph, const std::string& out_graph,
              const std::string& target_node) {
  GraphDef gdef;
  Status s;
  s = ReadTextProto(Env::Default(), in_graph, &gdef);
  if (!s.ok()) LOG(FATAL) << "Loading graphdef failed: " << s.error_message();

  auto hmod = ExtractHloFromGraphDef(std::move(gdef), target_node);

  s = WriteTextProto(Env::Default(), out_graph, hmod);
  if (!s.ok()) LOG(FATAL) << "Couldn't write hlo module: " << s.error_message();
  LOG(INFO) << "ALL DONE";
}

}


int main(int argc, char** argv) {

  std::string in_graph = "";
  std::string out_graph = "xla_module.pbtxt";
  std::string target_node = "";

  std::vector<tensorflow::Flag> flag_list = {
      {"in_graph", &in_graph, "input graph file name"},
      {"out_graph", &out_graph, "output graph def"},
      {"target_node", &target_node,
       "space separated list of target nodes for capture"}};

  std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  bool parsed_flags_ok = tensorflow::Flags::Parse(&argc, argv, flag_list);

  tensorflow::port::InitMain(argv[0], &argc, &argv);

  if (!parsed_flags_ok) {
    LOG(ERROR) << usage;
    return -1;
  }
  if (in_graph.empty()) {
    LOG(ERROR) << "in_graph graph can't be empty.\n" << usage;
    return -1;
  }
  if (target_node.empty()) {
    LOG(ERROR) << "target_node can't be empty.\n" << usage;
    return -1;
  }
  tensorflow::RealMain(in_graph, out_graph, target_node);
  return 0;
}