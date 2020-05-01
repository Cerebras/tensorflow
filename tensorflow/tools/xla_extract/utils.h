#pragma once

#define NO_LOG 0
#define INFO_LOG 1
#define DEBUG_LOG 2

namespace tensorflow {
namespace wse {

inline bool is_true(const char *s) {
  if (s && *s) {
    const char c = ::tolower(*s);
    if (c == 'y' || c == 't') {
      return true;
    }
    return atoi(s) > 0;
  }
  return false;
}

inline bool get_env_bool(const char *s, const bool dflt) {
  const char *v = getenv(s);
  if (v && *v) {
    return is_true(v);
  }
  return dflt;
}

inline int get_env_int(const char *s, const int dflt) {
  const char* v = getenv(s);
  if (v && *v) {
    return atoi(v);
  }
  return dflt;
}

template <typename MSG>
inline std::string msg_to_json(const MSG& msg) {
  std::string json;
  google::protobuf::util::JsonPrintOptions op;
  op.add_whitespace = true;
  google::protobuf::util::MessageToJsonString(msg, &json, op);
  return std::move(json);
}

template <typename MSG>
inline bool save_msg(const MSG& msg, const std::string& file) {
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

extern const bool save_messages;
extern const bool verbose;

}
}
