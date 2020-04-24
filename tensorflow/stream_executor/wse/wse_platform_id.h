/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_WSE_WSE_PLATFORM_ID_H_
#define TENSORFLOW_STREAM_EXECUTOR_WSE_WSE_PLATFORM_ID_H_

#include "tensorflow/stream_executor/platform.h"

namespace stream_executor {
namespace wse {

// Opaque and unique identifier for the WSE platform.
// This is needed so that plugins can refer to/identify this platform without
// instantiating a WsePlatform object.
// This is broken out here to avoid a circular dependency between WsePlatform
// and WseExecutor.
extern const Platform::Id kWsePlatformId;

}  // namespace rocm
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_WSE_WSE_PLATFORM_ID_H_
