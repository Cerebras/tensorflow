#pragma once

#include "tensorflow/compiler/xla/client/client_library.h"

namespace tensorflow {
namespace wse {

xla::StatusOr<std::unique_ptr<xla::HloModule>> RunHlo(std::unique_ptr<xla::HloModule>& hlo_module);

}
}
