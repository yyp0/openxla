#ifndef XLA_SERVICE_SPMD_ALPA_COMPILER_H_
#define XLA_SERVICE_SPMD_ALPA_COMPILER_H_

#include "xla/service/hlo_pass_pipeline.h"
#include "xla/pjrt/pjrt_client.h"

namespace xla {
namespace spmd {

// Run the auto sharding pass to add sharding anotations
// for each HLO instruction.
Status RunAutoShardingPass(HloModule* hlo_module, HloPassPipeline& /*,
                           const CompileOptions& options*/);

// Run the SPMD partitioner pass.
Status RunSpmdPartitionerPass(HloModule* hlo_module, HloPassPipeline& /*,
                              const CompileOptions& options*/);

// Set the shardings for output tensors.
Status SetHloModuleOutputShardings(HloModule* hlo_module,
                                   const std::vector<OpSharding>& op_shardings);

// Set the shardings for input tensors.
Status SetHloModuleInputShardings(HloModule* hlo_module,
                                  const std::vector<OpSharding>& op_shardings);

};  // namespace spmd
};  // namespace xla

#endif  // XLA_SERVICE_SPMD_ALPA_COMPILER_H_
