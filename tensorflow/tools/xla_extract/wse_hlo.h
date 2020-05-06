#ifndef WSE_HLO_LIB_H
#define WSE_HLO_LIB_H

#include "tensorflow/compiler/xla/client/client_library.h"

namespace tensorflow {
namespace wse {

xla::StatusOr<std::unique_ptr<xla::HloModule>> RunHlo(std::unique_ptr<xla::HloModule>& hlo_module);

}
}

#endif