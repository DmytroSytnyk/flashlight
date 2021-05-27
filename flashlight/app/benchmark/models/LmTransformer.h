/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/modules.h"
#include "flashlight/fl/flashlight.h"
#include "flashlight/fl/nn/modules/modules.h"

namespace fl {
namespace app {
namespace benchmark {

/**
 * This is example of plugin for language model architecture which is expected
 * the input with size Time x Batch x 1 x 1 and used with the adaptive softmax
 * as criterion (so the last linear layer is absent here)
 * This architecture is using also adaptive embedding and sinusoidal positional
 * embedding.
 */
class LmTransformer : public fl::Container {
 public:
  LmTransformer(int64_t nLabel, bool fp16 = false);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& input) override;

  std::string prettyString() const override;

 private:
  LmTransformer() = default;

  bool fp16_;

  std::shared_ptr<fl::Sequential> frontend_;
  std::vector<std::shared_ptr<fl::Transformer>> transformers_;
};

} // namespace benchmark
} // namespace app
} // namespace fl
