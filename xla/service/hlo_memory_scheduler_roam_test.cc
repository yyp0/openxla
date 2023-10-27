#include "xla/service/hlo_memory_scheduler.h"

#include <memory>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/heap_simulator.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_ordering.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {
namespace {
    const char* module_str = R"(
        HloModule test_aliasing_module

        ENTRY root {
        param = s32[1000] parameter(0)
        p0 = s32[1000] copy(param)
        p1 = s32[1000] copy(param)
        t = (s32[1000], s32[1000]) tuple(p0, p1)
        a = s32[1000] get-tuple-element(t), index=0
        b = s32[1000] get-tuple-element(t), index=1
        c = s32[1000] add(a, b)
        d = s32[1000] add(c, b)
        e = s32[1000] add(c, c)
        f = s32[1000] add(e, e)
        ROOT result = (s32[1000], s32[1000], s32[1000]) tuple(d, e, f)
        })";

    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(module_str));
    }

    
    TF_ASSIGN_OR_RETURN(std::unique_ptr<TuplePointsToAnalysis> points_to_analysis,
                      TuplePointsToAnalysis::Run(computation->parent()));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(computation->parent()));
    auto size_fn = [](const BufferValue& buffer) {
            return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
        };
    absl::flat_hash_map<const HloComputation*, int64_t> memory_by_computation;


    TF_CHECK_OK(RoamMemoryScheduler(
        HloCompuataion* computation,
        const TuplePointsToAnalysis& points_to_analysis,
        const HloAliasAnalysis& alias_analysis,
        const LogicalBuffer::SizeFunction& size_function,
        const absl::flat_hash_map<const HloComputation*, int64_t>&
            memory_by_computation
    ));
}