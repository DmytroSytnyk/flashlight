/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/benchmark/Utils.h"

#include "flashlight/fl/flashlight.h"

namespace fl {
namespace app {
namespace benchmark {

void initStates(std::string&& name, bool fp16) {
  name += fp16 ? " + AMP" : "";
  if (fl::getWorldRank() == 0) {
    std::cout << "\n----- " + name + " -----" << std::endl;
  }

  af::deviceGC();
  fl::DynamicBenchmark::setBenchmarkMode(true);
  fl::OptimMode::get().setOptimLevel(fl::OptimLevel::DEFAULT);
}

void printInfo(
    const fl::app::benchmark::ModelBenchmarker& benchmarker,
    int numUnits,
    bool verbose) {
  if (fl::getWorldRank() != 0) {
    return;
  }

  std::cout << "Throughput: "
            << fl::lib::format(
                   "%.2f",
                   numUnits * fl::getWorldSize() / benchmarker.getBatchTime());
  std::cout << std::endl;

  if (verbose) {
    std::cout << "\nBatch Time(ms): "
              << fl::lib::format("%.2f", benchmarker.getBatchTime() * 1000);
    std::cout << "\nModel Forward Time(ms): "
              << fl::lib::format("%.2f", benchmarker.getForwardTime() * 1000);
    std::cout << "\nCriterion Forward Time(ms): "
              << fl::lib::format("%.2f", benchmarker.getCriterionTime() * 1000);
    std::cout << "\nBackward Time(ms): "
              << fl::lib::format("%.2f", benchmarker.getBackwardTime() * 1000);
    std::cout << "\nOptimization Time(ms): "
              << fl::lib::format(
                     "%.2f", benchmarker.getOptimizationTime() * 1000);
    std::cout << std::endl;

    auto* curMemMgr =
        fl::MemoryManagerInstaller::currentlyInstalledMemoryManager();
    if (curMemMgr) {
      curMemMgr->printInfo("Memory Manager Stats", 0 /* device id */);
    }
  }
}

} // namespace benchmark
} // namespace app
} // namespace fl
