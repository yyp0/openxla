/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/hlo_memory_scheduler.h"

#include "pybind11/stl.h"

#include <algorithm>
#include <limits>
#include <map>
#include <queue>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/heap_simulator.h"
#include "xla/service/tuple_points_to_analysis.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/lib/gtl/map_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace {

using ::tsl::strings::HumanReadableNumBytes;
namespace py = pybind11;

class RoamScheduler {
  public:
    static StatusOr<HloInstructionSequence> Run(
        HloComputation* computation,
        const TuplePointsToAnalysis& points_to_analysis,
        const BufferValue::SizeFunction& size_function,
        const absl::flat_hash_map<const HloComputation*, int64_t>&
            memory_by_computation) {
      RoamScheduler scheduler(computation, points_to_analysis, size_function, 
                              memory_by_computation);

      return scheduler.CreateSchedule();
    }

  private:
    RoamScheduler(HloComputation* computation,
                  const TuplePointsToAnalysis& points_to_analysis,
                  const BufferValue::SizeFunction& size_function,
                  const absl::flat_hash_map<const HloComputation*, int64_t>& 
                    memory_by_computation)
        : computation_(computation),
          points_to_analysis_(points_to_analysis),
          size_function_(size_function),
          memory_by_computation_(memory_by_computation) {
      // Create a map containing the LogicalBuffer uses for each HLO
      // instruction. An HLO instruction "uses" a LogicalBuffer if the
      // LogicalBuffer is in an operand of the instruction as indicated by
      // points-to analysis.
      for (auto* instruction : computation->instructions()) {
        absl::flat_hash_set<const LogicalBuffer*> instr_uses;
        for (auto* operand : instruction->operands()) {
          points_to_analysis.GetPointsToSet(operand).ForEachElement(
              [&](const ShapeIndex& /*index*/,
                  const PointsToSet::BufferList& buffers) {
                instr_uses.insert(buffers.begin(), buffers.end());
              });
        }
        buffer_uses_[instruction] = std::vector<const LogicalBuffer*>(
            instr_uses.begin(), instr_uses.end());
      }

      // Create map containing the number of unscheduled uses (hlo instructions)
      // of each logical buffer.
      unscheduled_use_count_.reserve(points_to_analysis.num_logical_buffers());
      for (auto* instruction : computation->instructions()) {
        for (auto* buffer :
            points_to_analysis.GetBuffersDefinedByInstruction(instruction)) {
          unscheduled_use_count_[buffer] = 0;
        }
      }
      for (auto* instruction : computation->instructions()) {
        for (const LogicalBuffer* buffer : buffer_uses_.at(instruction)) {
          ++unscheduled_use_count_[buffer];
        }
      }

      // Buffers live out of the computation have an implicit use at the end of
      // the computation.
      for (const LogicalBuffer* live_out_buffer :
          points_to_analysis.GetPointsToSet(computation->root_instruction())
              .CreateFlattenedSet()) {
        ++unscheduled_use_count_[live_out_buffer];
      }
    }
    
    HloComputation* computation_;
    const TuplePointsToAnalysis& points_to_analysis_;
    const BufferValue::SizeFunction& size_function_;
    const absl::flat_hash_map<const HloComputation*, int64_t> memory_by_computation_;
    
    // Support ListMemory Scheduler for instructions scheduled at the same time in ILP results.
    // A map containing the LogicalBuffers that each instruction uses.
    absl::flat_hash_map<const HloInstruction*, std::vector<const LogicalBuffer*>>
        buffer_uses_;
    // A map containing the count of unscheduled HLOs which using a particular
    absl::flat_hash_map<const LogicalBuffer*, int64_t> unscheduled_use_count_;

    absl::flat_hash_map<const HloInstruction*, int64_t> asap_;
    absl::flat_hash_map<const HloInstruction*, int64_t> alap_;
    std::vector<std::tuple<int64_t, HloInstruction*>> memory_insensitive_points_; 

    absl::flat_hash_map<const LogicalBuffer*, int64_t> logical_buffer_unique_id_;
    absl::flat_hash_map<int64_t, const LogicalBuffer*> unique_id_logical_buffer_;
    std::vector<std::vector<std::vector<int64_t>>> max_useful_liverange_;
    std::vector<std::vector<std::vector<int64_t>>> dependencies_;
    std::vector<int64_t> logical_buffer_size_;
    std::vector<int64_t> persistent_buffer_;
    std::vector<int64_t> buffer_alias_;
    std::vector<std::vector<int64_t>> no_succ_buffer_;
    int64_t min_peak_memory;
    int64_t max_memory_usage;


    // Compute ASAP time of nodes.
    int64_t ComputeASAPTime(HloInstruction*);
    int64_t ComputeASAPMSTime(HloInstruction*);

    // Compute ALAP time of nodes.
    int64_t ComputeALAPTime(HloInstruction*);
    int64_t ComputeALAPMSTime(HloInstruction*, int64_t);

    // Update memory_insensitive_points_.
    Status SearchMemoryInsensitivePoints();

    Status ExtractGraphInformation();
    
    std::map<int64_t, std::vector<int64_t>> CallSolver();

    HloInstructionSequence CreateSchedule();
};


HloInstructionSequence RoamScheduler::CreateSchedule() {
  // Get necessary information needed by ILP Solver.
  VLOG(2) << "Step 1: extract graph information.";
  TF_CHECK_OK(ExtractGraphInformation());
  
  // Call Solver.
  VLOG(2) << "Step 2: call the solver.";
  std::map<int64_t, std::vector<int64_t>> created_time = CallSolver();

  
  // Transfer the created time to instruction sequence.
  VLOG(2) << "Step 3: start to generate schedule sequence from the solver result.";
  std::list<HloInstruction*> list_instructions;
  absl::flat_hash_map<HloInstruction*, int64_t> scheduled_instructions;
  int64_t schedule_index = 0;
  for (auto op : created_time) {
    int64_t time = op.first;
    std::vector<int64_t> buffer_ids = op.second;

    VLOG(1) << "Begin to analyze the priority of instructions at timestep " 
            << time << ".";
    // Sort instructions according to the priority of instruction.
    std::multimap<int64_t, HloInstruction*> undecided_instructions;
    for (auto id : buffer_ids) {
      auto curr_buffer = unique_id_logical_buffer_[id];
      HloInstruction* source = curr_buffer->instruction();
      
      int64_t defined_bytes = 0;
      auto defined_buffers = points_to_analysis_.GetBuffersDefinedByInstruction(source);
      for (auto buffer : defined_buffers) {
        defined_bytes += size_function_(*buffer);
      }

      int64_t freed_bytes = 0;
      for (auto operand : source->operands()) {
        points_to_analysis_.GetPointsToSet(operand).ForEachElement(
          [&] (const ShapeIndex&,
              const PointsToSet::BufferList& buffers) {
            for (auto buffer : buffers) {
              if (unscheduled_use_count_[buffer] == 0) {
                freed_bytes += size_function_(*buffer);
              }
            }
          }
        );
      }

      // undecided_instructions[freed_bytes - defined_bytes] = source;
      undecided_instructions.emplace(freed_bytes - defined_bytes, source);

      VLOG(3) << "The priority of instruction " << source->ToShortString() << " is " << freed_bytes - defined_bytes;
    }

    while (undecided_instructions.size() > 0) {
      auto best = undecided_instructions.end();
      --best;
      HloInstruction* best_instr = best->second;
      if (scheduled_instructions.find(best_instr) != 
          scheduled_instructions.end()) {
        undecided_instructions.erase(best);
        continue;
      }

      list_instructions.push_back(best_instr);
      scheduled_instructions[best_instr] = schedule_index++;
      undecided_instructions.erase(best);
    }
  }

  // Insert element-wise instructions into the sequence.
  std::vector<HloInstruction*> post_order = computation_->MakeInstructionPostOrder();
  for (int i = post_order.size() - 1; i >= 0; --i) {
    HloInstruction* curr_instr = post_order[i];

    if (scheduled_instructions.find(curr_instr) != 
        scheduled_instructions.end()) {
      continue;
    }

    // Get min index of users.
    int64_t min_index = INT_MAX;
    HloInstruction* insert_instr = nullptr;
    for (auto user : curr_instr->users()) {
      if (scheduled_instructions.find(user) == 
          scheduled_instructions.end()) {
        continue;
      }

      if (scheduled_instructions[user] < min_index) {
        min_index = scheduled_instructions[user];
        insert_instr = user;

        VLOG(3) << "Change the earliest user of instruction " 
                << curr_instr->ToShortString() << " to " << user->ToShortString()
                << " at timestep " << min_index;
      }
    }

    if (insert_instr != nullptr) {
      VLOG(2) << "Insert instruction " << curr_instr->ToShortString() 
            << " at timestep " << min_index
            << " before " << insert_instr->ToShortString();
      
      auto it = std::next(list_instructions.begin(), min_index);
      list_instructions.insert(it, curr_instr);
      scheduled_instructions[curr_instr] = min_index;
      
      for (auto& item : scheduled_instructions) {
        if (item.second >= min_index and item.first != curr_instr) {
          ++item.second;
        }
      }
    } else {
      VLOG(1) << "Not inserted instruction: " << curr_instr->ToShortString();
    }
  }

  int64_t schedule_time = 0;
  HloInstructionSequence schedule;
  for (auto instruction : list_instructions) {
    VLOG(1) << "Schedule " << instruction->ToShortString() << " at timestep " << schedule_time++;
    schedule.push_back(instruction);
  }
  CHECK_EQ(schedule.size(), computation_->instruction_count());

  return schedule;
}


std::map<int64_t, std::vector<int64_t>> RoamScheduler::CallSolver() {
  std::map<int64_t, std::vector<int64_t>> created;

  PyGILState_STATE gstate = PyGILState_Ensure();
  {
    // Import roam.
    VLOG(1) << "Begin to launch roam solver.";
    py::object submodule = py::module_::import("roam.scheduler_optimization");
    VLOG(1) << "Success to import roam.";
    py::object call_solver_serialized = submodule.attr("call_solver_serialized");
    VLOG(1) << "Success to get call_solver_serialized.";

    py::object ret = call_solver_serialized(
                      max_useful_liverange_, 
                      dependencies_,
                      logical_buffer_size_,
                      no_succ_buffer_,
                      min_peak_memory,
                      max_memory_usage,
                      persistent_buffer_,
                      buffer_alias_);
    
    VLOG(1) << "Finish execution of call_solver_serialized.";

    if (ret.is_none()) {
      PyGILState_Release(gstate);
      VLOG(1) << "No valid result generated by RoamScheduler.";
      exit(-1);
    }

    VLOG(1) << "Start to analyze the py result.";
    py::dict solver_res = ret.cast<py::dict>();
    for (const auto& item : solver_res) {
      int64_t id = item.first.cast<int64_t>();
      int64_t time = item.second.cast<int64_t>();
      created[time].push_back(id); 
    }
    VLOG(1) << "Finish the analysis of py result.";
  }
  PyGILState_Release(gstate);

  return created;  
}


Status RoamScheduler::ExtractGraphInformation() {
  // Obtain asap/alap time of all instructions.
  // int64_t num_instruction = computation_->instruction_count();
  HloInstruction* root = computation_->root_instruction();
  ComputeASAPMSTime(root);
  VLOG(1) << "ASAPMS time of root instruction: " << asap_[root];
  int64_t max_timesteps = asap_[root];
  for (auto instruction : computation_->instructions()) {
    // if (instruction->opcode() == HloOpcode::kParameter || 
    //     instruction->opcode() == HloOpcode::kConstant) {
    //   continue;
    // }

    if (asap_.find(instruction) == asap_.end()) {
      ComputeASAPMSTime(instruction);
    }
    
    if (alap_.find(instruction) == asap_.end()) {
      ComputeALAPMSTime(instruction, max_timesteps);
    }
    // asap_[instruction] = ComputeASAPTime(instruction);
    // alap_[instruction] = num_instruction - ComputeALAPTime(instruction) + 1;
  
    VLOG(1) << "HloInstruction " << instruction->ToShortString()
            << ": asap=" << asap_[instruction]
            << " alap=" << alap_[instruction];
  }

  // Get the number of logical buffers in the current computation.
  int64_t num_logical_buffers = 0;
  for (HloInstruction* curr_instr : computation_->instructions()) {
    auto fanouts = points_to_analysis_.GetBuffersDefinedByInstruction(curr_instr);
    for (auto buffer : fanouts) {
      VLOG(1) << "Init buffer: " << buffer->ToString() << " with id=" << num_logical_buffers;
      logical_buffer_unique_id_[buffer] = num_logical_buffers;
      unique_id_logical_buffer_[num_logical_buffers++] = buffer;
    }
  }

  // Init max_useful_liverange_ and dependencies_.
  max_useful_liverange_.resize(num_logical_buffers, 
                               std::vector<std::vector<int64_t>>(/*dynamic length*/));
  dependencies_.resize(num_logical_buffers, 
                       std::vector<std::vector<int64_t>>(3, 
                      std::vector<int64_t>(/*dynamic length*/)));
  logical_buffer_size_.resize(num_logical_buffers, -1);
  min_peak_memory = 0;
  max_memory_usage = 0;
  VLOG(1) << "The total number of logical buffers in the current computation: " << num_logical_buffers;


  absl::flat_hash_set<int64_t> traversed_buffer;
  for (HloInstruction* curr_instr : computation_->instructions()) {
    int64_t required_memory = 0;
    auto succesors = points_to_analysis_.GetBuffersDefinedByInstruction(curr_instr);
    for (auto succ : succesors) {
      int64_t succ_size = size_function_(*succ);
      max_memory_usage += succ_size;
      required_memory += succ_size;
      int64_t succ_id = logical_buffer_unique_id_[succ];
      logical_buffer_size_[succ_id] = succ_size;
    }
    
    absl::flat_hash_set<const LogicalBuffer*> used_buffers;
    for (auto operand : curr_instr->operands()) {
      points_to_analysis_.GetPointsToSet(operand).ForEachElement(
        [&] (const ShapeIndex&,
            const PointsToSet::BufferList& buffers) {
              used_buffers.insert(buffers.begin(), buffers.end());
        }
      );
    }

    int64_t asap_r, alap_r;
    asap_r = asap_[curr_instr];
    alap_r = alap_[curr_instr];
    std::vector<int64_t> used_buffers_id;
    for (auto buffer : used_buffers) {
      if (curr_instr == computation_->root_instruction()) {
        VLOG(1) << "Operand of root instruction: " << buffer->ToString();
      }

      int64_t curr_buffer_id = logical_buffer_unique_id_[buffer];
      used_buffers_id.emplace_back(curr_buffer_id);
      int64_t buffer_size = size_function_(*buffer);
      required_memory += buffer_size;
      
      // TODO(HuiyaoSHU.HYS): feat the alias information.
      // Get buffer alias information.
      // auto alias = points_to_analysis_.GetBufferAliases(*buffer);
      // for (auto alias_buffer : alias) {
      //   if (traversed_buffer.find(alias_buffer) != 
      //       traversed_buffer.end()) {
      //     buffer_alias_[curr_buffer_id].push_back(
      //                    logical_buffer_unique_id_[alias_buffer]);
      //   }
      // }

      // Get source of the logical buffer.
      HloInstruction* source = buffer->instruction();
      if (source->opcode() == HloOpcode::kParameter) {
        persistent_buffer_.push_back(logical_buffer_unique_id_[buffer]);
      }

      int64_t asap_l, alap_l;
      asap_l = asap_[source];
      alap_l = alap_[source];

      // Update max useful liverange for each buffer.
      if (max_useful_liverange_[curr_buffer_id].size() == 0) {
        max_useful_liverange_[curr_buffer_id].push_back({asap_l, alap_l});
      }
      VLOG(1) << "Buffer id: " << curr_buffer_id << " snk: " << curr_instr->ToShortString();
      max_useful_liverange_[curr_buffer_id].push_back({asap_r, alap_r});

      // Get the siblings logical buffers.
      auto siblings = points_to_analysis_.GetBuffersDefinedByInstruction(source);
      for (auto sib : siblings) {
        if (traversed_buffer.find(curr_buffer_id) != 
            traversed_buffer.end()) {
          break;
        } 

        int64_t sib_id = logical_buffer_unique_id_[sib];
        dependencies_[curr_buffer_id][2].push_back(sib_id);
      }
      traversed_buffer.insert(curr_buffer_id);

      // Get the succesors logical buffers.
      for (auto succ : succesors) {
        int64_t succ_size = size_function_(*succ);

        int64_t succ_id = logical_buffer_unique_id_[succ];
        if (succ->instruction() == computation_->root_instruction()) {
          HloInstruction* root = computation_->root_instruction();
          if (max_useful_liverange_[succ_id].size() == 0) {
            max_useful_liverange_[succ_id].push_back({asap_[root], alap_[root]});
            max_useful_liverange_[succ_id].push_back({alap_[root], alap_[root]});
          }
        }
        dependencies_[curr_buffer_id][1].push_back(succ_id);
        dependencies_[succ_id][0].push_back(curr_buffer_id);
      }
    }
    
    // Temporarily hardcode for node without fanouts.
    if (succesors.size() == 0 and 
        curr_instr == computation_->root_instruction()) {
      VLOG(1) << "Instruction: " << curr_instr->ToShortString() << " has no successors.";
      
      used_buffers_id.emplace_back(alap_[curr_instr]);
      no_succ_buffer_.emplace_back(used_buffers_id);  
    }
    
    if (required_memory > min_peak_memory) {
      min_peak_memory = required_memory;
    }
  }

  return OkStatus();
}

// Need to be optimized.
int64_t RoamScheduler::ComputeASAPTime(HloInstruction* instruction) {
  absl::flat_hash_set<HloInstruction*> traversed;
  std::vector<HloInstruction*> BFS = {instruction};
  int64_t traversed_param = 0;

  while (BFS.size() > 0) {
    HloInstruction* top = BFS.back();
    BFS.pop_back();
    if (top->opcode() == HloOpcode::kParameter or \
        traversed.find(top) != traversed.end()) {
      continue;
    }
    traversed.emplace(top);
    
    if (top->opcode() == HloOpcode::kParameter || 
        top->opcode() == HloOpcode::kConstant) {
      traversed_param += 1;
    }

    // Need to take control_precedence and control_successors.
    auto operands = top->operands();
    for (auto instr : operands) {
      BFS.push_back(instr);
    } 
  }

  return traversed.size() - traversed_param;
}

int64_t RoamScheduler::ComputeASAPMSTime(HloInstruction* instruction) {
  // VLOG(1) << "Compute ASAPMS time for instruction: " << instruction->ToShortString();
  if (asap_.find(instruction) != asap_.end()) {
    // VLOG(1) << "Get existed ASAPMS value: " << asap_[instruction] 
    //         << " of instruction: " << instruction->ToShortString();

    return asap_[instruction];
  }

  int64_t time = 1;
  // TODO(HuiyaoShu): validate the operands function.
  auto operands = instruction->operands();
  for (auto ope : operands) {
    // VLOG(1) << "Operand: " << ope->ToShortString() 
    //         << " of instruction: " << instruction->ToShortString();
    int64_t t = ComputeASAPMSTime(ope);
    time = std::max(t + 1, time);
  }

  asap_[instruction] = time;

  return time;
}

int64_t RoamScheduler::ComputeALAPTime(HloInstruction* instruction) {
  absl::flat_hash_set<HloInstruction*> traversed;
  std::vector<HloInstruction*> BFS = {instruction};
  int64_t traversed_param = 0;

  while(BFS.size() > 0) {
    HloInstruction* top = BFS.back();
    BFS.pop_back();
    if (traversed.find(top) != traversed.end()) {
      continue;
    }
    traversed.emplace(top);

    if (top->opcode() == HloOpcode::kParameter || 
        top->opcode() == HloOpcode::kConstant) {
      traversed_param += 1;
    }

    std::vector<HloInstruction*> users = top->users();
    for (auto instr : users) {
      BFS.push_back(instr);
    }
  }

  return traversed.size() - traversed_param;
}

int64_t RoamScheduler::ComputeALAPMSTime(HloInstruction* instruction, int64_t max_timesteps) {
  // VLOG(1) << "Compute ALAPMS time for instruction: " << instruction->ToShortString()
  //         << " Max timesteps: " << max_timesteps;
  if (alap_.find(instruction) != alap_.end()) {
    return alap_[instruction];
  }

  int64_t time = max_timesteps;
  auto users = instruction->users();
  for (auto user : users) {
    int64_t t = ComputeALAPMSTime(user, max_timesteps);
    time = std::min(time, t - 1);
  }

  alap_[instruction] = time;

  return time;
}

Status RoamScheduler::SearchMemoryInsensitivePoints() {
  int64_t num_instr = computation_->instruction_count();
  VLOG(1) << "The total number of instructions in the computation: " << num_instr;

  int64_t total = 0;
  for (auto instruction : computation_->instructions()) {
    if (instruction->opcode() == HloOpcode::kParameter || 
        instruction->opcode() == HloOpcode::kConstant) {
      continue;
    }
    total += 1;
  }
  VLOG(1) << "Instruction except parameter: " << total;

  // Compute asap/alap time for instructions.
  for (auto instruction : computation_->instructions()) {
    if (instruction->opcode() == HloOpcode::kParameter || 
        instruction->opcode() == HloOpcode::kConstant) {
      continue;
    }

    asap_[instruction] = ComputeASAPTime(instruction);
    alap_[instruction] = total - ComputeALAPTime(instruction) + 1;
  
    VLOG(1) << "HloInstruction " << instruction->ToShortString()
            << ": asap=" << asap_[instruction]
            <<  " alap=" << alap_[instruction];
  }
   
  for (auto instruction : computation_->instructions()) {
    if (asap_[instruction] == alap_[instruction]) {
      memory_insensitive_points_.emplace_back(asap_[instruction], instruction);
      // memory_insensitive_points_[instruction] = asap[instruction];
    }
  }

  return OkStatus();
}

// Class implementing a list scheduler of HLO instructions which produces a
// sequence which minimizes memory usage by preferring to schedule the node that
// frees bigger buffer and defines smaller outputs.
//
// Note that list scheduler is a greedy algorithm which cannot guarantee a
// global optimal solution. As a counterexample, considering the following
// graph:
//
//      +--> B ===> C -------+
// A -> |                    |
//      |                    v
//      +--> D ---> F=======>G
//      |           ^
//      |           |
//      +--> E -----+
//
//  --> : Buffer with size 1
//  ==> : Buffer with size 2
//
// The list scheduler will always try to defer scheduling B in a greedy way
// since its output buffer is bigger than input. The sequence it creates will
// be:
//   A D E F B C G
// , which has a maximum memory usage of 6 (B is alive while F is executing).
//
// An optimal way to schedule the previous graph is:
//   A B C D E F G
// , which has a maximum memory usage of 5 (when F is executing).
//
class ListScheduler {
 public:
  // Construct and return a memory-minimizing sequence of HLO instructions
  // containing the given HLO computation.
  static StatusOr<HloInstructionSequence> Run(
      HloComputation* computation,
      const TuplePointsToAnalysis& points_to_analysis,
      const BufferValue::SizeFunction& size_function,
      const absl::flat_hash_map<const HloComputation*, int64_t>&
          memory_by_computation) {
    ListScheduler scheduler(computation, points_to_analysis, size_function,
                            memory_by_computation);
    return scheduler.CreateSchedule();
  }

  // Returns whether the memory used by the given HLO should be ignored by the
  // scheduling heuristic.
  static bool IgnoreInstruction(const HloInstruction& instruction) {
    return instruction.opcode() == HloOpcode::kParameter ||
           instruction.opcode() == HloOpcode::kConstant;
  }

 private:
  // The scheduling priority of an instruction is first the number of bytes
  // freed by scheduling the instruction, and second (tie-breaker) by the number
  // of users. This is represented as a std::pair containing these two values
  // (first element is the bytes freed). std::pair provides the necessary
  // comparison operators.
  using Priority = std::pair<int64_t, int64_t>;

  ListScheduler(HloComputation* computation,
                const TuplePointsToAnalysis& points_to_analysis,
                const BufferValue::SizeFunction& size_function,
                const absl::flat_hash_map<const HloComputation*, int64_t>&
                    memory_by_computation)
      : computation_(computation),
        points_to_analysis_(points_to_analysis),
        size_function_(size_function),
        memory_by_computation_(memory_by_computation) {
    // Create a map containing the LogicalBuffer uses for each HLO
    // instruction. An HLO instruction "uses" a LogicalBuffer if the
    // LogicalBuffer is in an operand of the instruction as indicated by
    // points-to analysis.
    for (auto* instruction : computation->instructions()) {
      absl::flat_hash_set<const LogicalBuffer*> instr_uses;
      for (auto* operand : instruction->operands()) {
        points_to_analysis.GetPointsToSet(operand).ForEachElement(
            [&](const ShapeIndex& /*index*/,
                const PointsToSet::BufferList& buffers) {
              instr_uses.insert(buffers.begin(), buffers.end());
            });
      }
      buffer_uses_[instruction] = std::vector<const LogicalBuffer*>(
          instr_uses.begin(), instr_uses.end());
    }

    // Create map containing the number of unscheduled uses (hlo instructions)
    // of each logical buffer.
    unscheduled_use_count_.reserve(points_to_analysis.num_logical_buffers());
    for (auto* instruction : computation->instructions()) {
      for (auto* buffer :
           points_to_analysis.GetBuffersDefinedByInstruction(instruction)) {
        unscheduled_use_count_[buffer] = 0;
      }
    }
    for (auto* instruction : computation->instructions()) {
      for (const LogicalBuffer* buffer : buffer_uses_.at(instruction)) {
        ++unscheduled_use_count_[buffer];
      }
    }

    // Buffers live out of the computation have an implicit use at the end of
    // the computation.
    for (const LogicalBuffer* live_out_buffer :
         points_to_analysis.GetPointsToSet(computation->root_instruction())
             .CreateFlattenedSet()) {
      ++unscheduled_use_count_[live_out_buffer];
    }
  }

  // Returns whether the memory used by the given buffer should be ignored by
  // the scheduling heuristic.
  static bool IgnoreBuffer(const LogicalBuffer& buffer) {
    return IgnoreInstruction(*buffer.instruction());
  }

  // An entry in the worklist used by CreateSchedule.  Corresponds to one
  // HloInstruction, plus some cached metadata, saved for the purposes of making
  // BytesFreedIfScheduled fast.
  struct ReadyListEntry {
    HloInstruction* instruction;

    // The total size of all buffers defined by this instruction.
    int64_t bytes_defined;

    // For each buffer B used by this instruction, we keep a pair (B, U), where
    // U is the number of uses of B that have not yet been scheduled. This pair
    // is a pointer into the unscheduled_use_count_ map, so it gets updated for
    // free when we update counts in the map.
    std::vector<const std::pair<const LogicalBuffer* const, int64_t>*>
        used_buffer_unscheduled_use_counts;
  };

  // Creates a ReadyListEntry for the given instruction.
  ReadyListEntry MakeReadyListEntry(HloInstruction* instruction) {
    ReadyListEntry entry;
    entry.instruction = instruction;

    entry.bytes_defined = 0;
    for (auto* buffer :
         points_to_analysis_.GetBuffersDefinedByInstruction(instruction)) {
      if (!IgnoreBuffer(*buffer)) {
        entry.bytes_defined += size_function_(*buffer);
      }
    }

    for (auto* buffer : buffer_uses_.at(instruction)) {
      if (IgnoreBuffer(*buffer)) {
        continue;
      }
      auto unscheduled_use_count_it = unscheduled_use_count_.find(buffer);
      CHECK(unscheduled_use_count_it != unscheduled_use_count_.end());
      entry.used_buffer_unscheduled_use_counts.push_back(
          &*unscheduled_use_count_it);
    }
    return entry;
  }

  // Returns the number of bytes freed *after* the HLO instruction finishes.
  // The current List algorithm only considers two states for an instruction:
  // right before it runs, and after it finishes. We don't represent memory
  // usage during the execution of an instruction. But if the instruction calls
  // subcomputations, they are only live during the instruction's execution.
  // We end up counting the memory used by subcomputations as memory "defined"
  // by the instruction. This is not entirely accurate, but it is more accurate
  // than not taking subcomputations into account at all. In the future, we may
  // improve accounting for subcomputation memory (b/65409243).
  int64_t BytesFreedIfScheduled(const ReadyListEntry& entry) {
    auto instruction = entry.instruction;
    auto opcode = instruction->opcode();

    // Scheduling the outfeed early and the infeed late gives more time to the
    // communicating processor to do its work.
    if (opcode == HloOpcode::kOutfeed &&
        !instruction->outfeed_config().empty()) {
      return INT_MAX;
    }
    if (opcode == HloOpcode::kInfeed && !instruction->infeed_config().empty()) {
      return INT_MIN;
    }

    int64_t freed_bytes = 0;
    for (const auto& kv : entry.used_buffer_unscheduled_use_counts) {
      auto buffer = kv->first;
      auto use_count = kv->second;
      if (use_count == 1) {
        freed_bytes += size_function_(*buffer);
      }
    }
    // We only count the memory usage of the largest subcomputation, instead of
    // adding them all, because subcomputations won't execute in parallel.
    int64_t max_subcomputation_bytes = 0;
    for (const auto* c : instruction->called_computations()) {
      auto it = memory_by_computation_.find(c);
      if (it != memory_by_computation_.end()) {
        int64_t subcomputation_bytes = it->second;
        if (subcomputation_bytes > max_subcomputation_bytes) {
          max_subcomputation_bytes = subcomputation_bytes;
        }
      }
    }
    int64_t bytes_defined;
    if (max_subcomputation_bytes > 0 &&
        (opcode == HloOpcode::kWhile || opcode == HloOpcode::kCall ||
         opcode == HloOpcode::kConditional)) {
      // The output buffer of while/call/conditional is always aliased with the
      // output buffer of the root instruction in the body. Don't double count.
      bytes_defined = max_subcomputation_bytes;
    } else {
      bytes_defined = entry.bytes_defined + max_subcomputation_bytes;
    }
    return freed_bytes - bytes_defined;
  }

  // Constructs the scheduling priority of the given instruction.
  Priority GetPriority(const ReadyListEntry& entry) {
    // Try to cluster scalars as close together as possible so that if they are
    // in unfused hlos, they can still live in machine registers without
    // excessive spilling.
    if (ShapeUtil::IsEffectiveScalar(entry.instruction->shape())) {
      return {std::numeric_limits<int64_t>::max(),
              std::numeric_limits<int64_t>::max()};
    }
    return {BytesFreedIfScheduled(entry), entry.instruction->user_count()};
  }

  HloInstructionSequence CreateSchedule() {
    HloInstructionSequence schedule;

    // Populate the ready list with instructions which have no operands or
    // control predecessors.
    absl::flat_hash_map<const HloInstruction*, int64_t> unscheduled_pred_count;
    for (auto* instruction : computation_->instructions()) {
      // TODO(b/34466113): Replace this and above with successors() or
      // predecessors() when these methods are added to HloInstruction.
      for (HloInstruction* user : instruction->users()) {
        unscheduled_pred_count[user]++;
      }
      for (HloInstruction* succ : instruction->control_successors()) {
        unscheduled_pred_count[succ]++;
      }
    }

    // Use a multimap to sort ReadyListEntry according to their priority.
    std::multimap<Priority, ReadyListEntry> ready_queue;

    // Map of ready instructions to their iterators in ready_queue.
    absl::flat_hash_map<const HloInstruction*,
                        std::multimap<Priority, ReadyListEntry>::iterator>
        ready_instructions;

    auto add_to_ready_queue = [&](HloInstruction* inst) {
      auto entry = MakeReadyListEntry(inst);
      auto it = ready_queue.emplace(GetPriority(entry), std::move(entry));
      ready_instructions[inst] = it;
    };

    for (auto* instruction : computation_->instructions()) {
      if (instruction->operands().empty() &&
          instruction->control_predecessors().empty()) {
        add_to_ready_queue(instruction);
      }
    }

    while (!ready_queue.empty()) {
      // Remove the selected instruction from the ready list and add it to the
      // schedule.
      auto best_it = ready_queue.end();
      --best_it;
      HloInstruction* best = best_it->second.instruction;
      VLOG(2) << "Schedule instruction: " << best->ToShortString()
              << " Bytes freed: " << best_it->first.first;
      ready_queue.erase(best_it);
      ready_instructions.erase(best);
      schedule.push_back(best);
      scheduled_instructions_.insert(best);

      bool adjust_ready_queue = false;
      // Update the unscheduled uses of the logical buffers.
      for (const LogicalBuffer* buffer : buffer_uses_.at(best)) {
        int64_t& count = unscheduled_use_count_[buffer];
        CHECK_GT(count, 0);
        --count;
        if (count == 1) {
          adjust_ready_queue = true;
        }
      }

      // Add new instructions to ready list.
      auto update_pred_count = [&](HloInstruction* inst) {
        int64_t pred_count = --unscheduled_pred_count.at(inst);
        CHECK_GE(pred_count, 0);
        if (pred_count == 0) {
          add_to_ready_queue(inst);
        }
      };
      // TODO(b/34466113): Replace this and above with successors() or
      // predecessors() when these methods are added to HloInstruction.
      for (HloInstruction* user : best->users()) {
        update_pred_count(user);
      }
      for (HloInstruction* succ : best->control_successors()) {
        update_pred_count(succ);
      }
      // The unscheduled use count for a buffer has changed to 1, so the
      // priorities of some ready instructions may go up. We update them in the
      // ready queue, so that they can appear earlier.
      if (adjust_ready_queue) {
        for (HloInstruction* operand : best->operands()) {
          for (HloInstruction* operand_user : operand->users()) {
            auto ready_instructions_it = ready_instructions.find(operand_user);
            if (ready_instructions_it == ready_instructions.end()) {
              continue;
            }
            auto ready_queue_it = ready_instructions_it->second;
            auto& entry = ready_queue_it->second;
            Priority new_priority = GetPriority(entry);
            if (new_priority == ready_queue_it->first) {
              continue;
            }
            // Create a new entry in ready_queue, then update
            // ready_instructions[operand_user] to refer to the new entry.
            ready_instructions_it->second =
                ready_queue.emplace(new_priority, std::move(entry));
            // Remove the old entry in ready_queue.
            ready_queue.erase(ready_queue_it);
          }
        }
      }
    }
    CHECK_EQ(schedule.size(), computation_->instruction_count());
    CHECK_EQ(scheduled_instructions_.size(), computation_->instruction_count());

    return schedule;
  }

  HloComputation* computation_;
  const TuplePointsToAnalysis& points_to_analysis_;
  const BufferValue::SizeFunction& size_function_;
  // Computations are analyzed in post-order. When scheduling an instruction
  // that includes subcomputations, such as a while loop, we use this map to
  // look up the memory needed by subcomputations.
  const absl::flat_hash_map<const HloComputation*, int64_t>&
      memory_by_computation_;

  // A map containing the LogicalBuffers that each instruction uses.
  absl::flat_hash_map<const HloInstruction*, std::vector<const LogicalBuffer*>>
      buffer_uses_;

  // A map containing the count of unscheduled HLOs which using a particular
  // LogicalBuffer.
  absl::flat_hash_map<const LogicalBuffer*, int64_t> unscheduled_use_count_;

  // Set of instructions which have been scheduled.
  absl::flat_hash_set<const HloInstruction*> scheduled_instructions_;
};

int64_t SumLogicalBufferSizes(
    const TuplePointsToAnalysis::BufferDefinitionVector& buffers,
    const BufferValue::SizeFunction& size_function) {
  int64_t size = 0;
  for (const LogicalBuffer* buffer : buffers) {
    size += size_function(*buffer);
  }
  return size;
}

StatusOr<HloInstructionSequence> ScheduleComputationHelper(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_function,
    const MemorySchedulerAlgorithm& algorithm,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        memory_by_computation,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory) {
  VLOG(2) << "Computation: " << computation->name();

  if (algorithm) {
    return algorithm(computation, points_to_analysis, alias_analysis,
                     size_function, memory_by_computation, postprocessor,
                     peak_memory);
  }
  return DefaultMemoryScheduler(computation, points_to_analysis, alias_analysis,
                                size_function, memory_by_computation,
                                postprocessor, peak_memory);
}

}  // namespace

StatusOr<HloInstructionSequence> DFSMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        memory_by_computation,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory) {
  // These variables are a hack to prevent overflows.
  int64_t cumulative_total_size = 0;
  int64_t total_hlos = computation->instruction_count();
  struct Stats {
    // Transitively includes the count of all nodes that lead to it.
    int64_t extra_users = 0;
    // Transitively includes the sizes of all nodes that lead to it.
    int64_t total_sizes = 0;
  };
  absl::flat_hash_map<const HloInstruction*, Stats> stats_map;
  stats_map.reserve(computation->instruction_count());

  for (const HloInstruction* hlo : computation->MakeInstructionPostOrder()) {
    auto& stats = stats_map[hlo];
    if (ListScheduler::IgnoreInstruction(*hlo)) {
      continue;
    }
    // This ordering is based on DFS post-order, with a heuristic to decide
    // which operand to visit first.  The heuristic is based on 'extra_users',
    // which is simply users-1 for each instruction.  By subtracting 1, we're
    // saying that instructions with no users or a single user don't count;
    // instructions with lots of fan-out will be visited earlier.
    stats.extra_users = hlo->users().empty() ? 0 : hlo->users().size() - 1;
    int64_t logical_buffer_size = SumLogicalBufferSizes(
        points_to_analysis.GetBuffersDefinedByInstruction(hlo), size_function);
    stats.total_sizes = logical_buffer_size;
    cumulative_total_size += logical_buffer_size;
    absl::flat_hash_set<const HloInstruction*> unique_operands(
        hlo->operands().begin(), hlo->operands().end());
    for (const HloInstruction* operand : unique_operands) {
      auto& operand_stats = stats_map.at(operand);
      stats.extra_users += operand_stats.extra_users;
      stats.total_sizes += operand_stats.total_sizes;
    }
    // stats.total_sizes transitively includes the sizes of all nodes that
    // lead to it. But computation is a DAG, so we are double-counting nodes,
    // which can lead to overflows for large programs.
    // cumulative_total_size caps the size to prevent overflows.
    // Same for total_hlos: it prevents overflows on very large and branchy
    // models, where the number of paths is exponential to the number of nodes.
    // NOTE(dimvar): this is quite ugly and should be changed. It's unclear
    // why we care about transitive sizes; when scheduling a node, its input
    // and output buffers should be all that matters, not its "history".
    stats.total_sizes = std::min(stats.total_sizes, cumulative_total_size);
    stats.extra_users = std::min(stats.extra_users, total_hlos);
  }
  CHECK_EQ(stats_map.size(), computation->instruction_count());

  // Construct a total order based on DFS post-order, visiting operands in
  // decreasing cumulative extra user order, and next by cumulative size, with a
  // tiebreaker by name for determinism.
  HloInstructionSequence sequence;
  FunctionVisitor visitor([&sequence](HloInstruction* hlo) {
    sequence.push_back(hlo);
    return OkStatus();
  });
  visitor.ReserveVisitStates(computation->instruction_count());
  TF_RETURN_IF_ERROR(computation->AcceptWithOperandOrder(
      &visitor, [&stats_map](const HloInstruction* a, const HloInstruction* b) {
        auto& stats_a = stats_map.at(a);
        auto& stats_b = stats_map.at(b);
        if (stats_a.extra_users != stats_b.extra_users) {
          return stats_a.extra_users > stats_b.extra_users;
        }
        if (stats_a.total_sizes != stats_b.total_sizes) {
          return stats_a.total_sizes > stats_b.total_sizes;
        }
        return a->name() < b->name();
      }));
  if (postprocessor) {
    sequence = postprocessor(sequence);
  }
  CHECK_EQ(sequence.size(), computation->instruction_count());
  if (peak_memory) {
    TF_ASSIGN_OR_RETURN(
        *peak_memory, HeapSimulator::MinimumMemoryForComputation(
                          *computation, sequence, alias_analysis, size_function,
                          &memory_by_computation));
  }
  return sequence;
}  // namespace xla

ModuleSchedulerAlgorithm ComputationSchedulerToModuleScheduler(
    const MemorySchedulerAlgorithm& computation_scheduler,
    const MemorySchedulerPostprocessor& postprocessor) {
  return [computation_scheduler, postprocessor](
             const HloModule* module,
             const TuplePointsToAnalysis& points_to_analysis,
             const HloAliasAnalysis& alias_analysis,
             const LogicalBuffer::SizeFunction& size_func,
             const absl::flat_hash_set<absl::string_view>& execution_threads,
             int64_t* peak_memory) -> StatusOr<HloSchedule> {
    HloSchedule schedule(module);
    absl::flat_hash_map<const HloComputation*, int64_t> memory_by_computation;
    for (auto* computation :
         module->MakeComputationPostOrder(execution_threads)) {
      if (!computation->IsFusionComputation()) {
        TF_ASSIGN_OR_RETURN(
            HloInstructionSequence computation_sequence,
            ScheduleComputationHelper(
                computation, points_to_analysis, alias_analysis, size_func,
                computation_scheduler, memory_by_computation, postprocessor,
                /*peak_memory=*/nullptr));
        schedule.set_sequence(computation, std::move(computation_sequence));
      }
    }
    if (peak_memory) {
      TF_ASSIGN_OR_RETURN(*peak_memory, HeapSimulator::MinimumMemoryForModule(
                                            schedule, size_func));
    }
    return std::move(schedule);
  };
}

StatusOr<HloInstructionSequence> RoamMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis, 
    const BufferValue::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        memory_by_computation,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory) {
  TF_ASSIGN_OR_RETURN(HloInstructionSequence sequence,
                      RoamScheduler::Run(computation, points_to_analysis, 
                                         size_function, memory_by_computation));

  if (postprocessor) {
    sequence = postprocessor(sequence);
  }

  if (peak_memory) {
    TF_ASSIGN_OR_RETURN(
        *peak_memory, HeapSimulator::MinimumMemoryForComputation(
                          *computation, sequence, alias_analysis, size_function,
                          &memory_by_computation));
  }
  
  return sequence;
}

StatusOr<HloInstructionSequence> ListMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        memory_by_computation,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory) {
  TF_ASSIGN_OR_RETURN(HloInstructionSequence sequence,
                      ListScheduler::Run(computation, points_to_analysis,
                                         size_function, memory_by_computation));
  
  // // Test RoamScheduler.
  // TF_CHECK_OK(RoamScheduler::Run(computation, points_to_analysis,
  //                                size_function, memory_by_computation));

  if (postprocessor) {
    sequence = postprocessor(sequence);
  }
  if (peak_memory) {
    TF_ASSIGN_OR_RETURN(
        *peak_memory, HeapSimulator::MinimumMemoryForComputation(
                          *computation, sequence, alias_analysis, size_function,
                          &memory_by_computation));
  }
  return sequence;
}

StatusOr<HloInstructionSequence> PostOrderMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        memory_by_computation,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory) {
  HloInstructionSequence sequence(computation->MakeInstructionPostOrder());
  if (postprocessor) {
    sequence = postprocessor(sequence);
  }
  if (peak_memory) {
    TF_ASSIGN_OR_RETURN(
        *peak_memory, HeapSimulator::MinimumMemoryForComputation(
                          *computation, sequence, alias_analysis, size_function,
                          &memory_by_computation));
  }
  return sequence;
}

StatusOr<HloInstructionSequence> DefaultMemoryScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64_t>&
        memory_by_computation,
    const MemorySchedulerPostprocessor& postprocessor, int64_t* peak_memory) {
  // Temporarily ignore when test alpa.
  // Support optimization with interger linea program method in full graph mode.
  // int64_t solver_memory = INT_MAX;
  // TF_ASSIGN_OR_RETURN(
  //     HloInstructionSequence solver_sequence,
  //     RoamMemoryScheduler(computation, points_to_analysis, alias_analysis,
  //                         size_function, memory_by_computation, postprocessor,
  //                         &solver_memory));

  // VLOG(1) << "Min-memory solver(full graph) sequence: " << HumanReadableNumBytes(solver_memory);

  // We try a few schedulers and choose whichever returns a lower min-memory,
  // not accounting for fragmentation.
  // - List is a scheduler that uses greedy heuristics.
  // - DFS visits HLOs in postorder, with a heuristic to decide the order of
  //   children.
  // - Postorder does not use any heuristics.
  // List wins for most of our benchmarks; postorder-based schedulers win for
  // some RNNs.
  int64_t list_memory;
  TF_ASSIGN_OR_RETURN(
      HloInstructionSequence list_sequence,
      ListMemoryScheduler(computation, points_to_analysis, alias_analysis,
                          size_function, memory_by_computation, postprocessor,
                          &list_memory));
  VLOG(1) << "Min-memory list sequence: " << HumanReadableNumBytes(list_memory);

  int64_t dfs_memory;
  TF_ASSIGN_OR_RETURN(
      HloInstructionSequence dfs_sequence,
      DFSMemoryScheduler(computation, points_to_analysis, alias_analysis,
                         size_function, memory_by_computation, postprocessor,
                         &dfs_memory));
  VLOG(1) << "Min-memory dfs sequence: " << HumanReadableNumBytes(dfs_memory);

  int64_t post_order_memory;
  TF_ASSIGN_OR_RETURN(
      HloInstructionSequence post_order_sequence,
      PostOrderMemoryScheduler(computation, points_to_analysis, alias_analysis,
                               size_function, memory_by_computation,
                               postprocessor, &post_order_memory));
  VLOG(1) << "Min-memory post order sequence: "
          << HumanReadableNumBytes(post_order_memory);

  auto min_memory = std::min({dfs_memory, post_order_memory, list_memory}); //, solver_memory});
  if (peak_memory) {
    *peak_memory = min_memory;
  }

  /*
  if (min_memory == solver_memory) {
    // Hardcode walkround for alpa test.
    HloInstructionSequence solver_sequence;
    VLOG(1) << "Chose min-memory solver sequence: "
            << HumanReadableNumBytes(solver_memory);
    return solver_sequence;
  }
  else */
  if (min_memory == list_memory) {
    VLOG(1) << "Chose min-memory list sequence: "
            << HumanReadableNumBytes(list_memory);
    return list_sequence;
  } else if (min_memory == dfs_memory) {
    VLOG(1) << "Chose min-memory dfs sequence: "
            << HumanReadableNumBytes(dfs_memory);
    return dfs_sequence;
  } else {
    VLOG(1) << "Chose min-memory post_order sequence: "
            << HumanReadableNumBytes(post_order_memory);
    return post_order_sequence;
  }
}

StatusOr<HloSchedule> DefaultModuleScheduler(
    const HloModule* module, const TuplePointsToAnalysis& points_to_analysis,
    const HloAliasAnalysis& alias_analysis,
    const BufferValue::SizeFunction& size_function,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    int64_t* peak_memory) {
  // We try a few schedulers and choose whichever returns a lower min-memory,
  // not accounting for fragmentation.
  // - List is a scheduler that uses greedy heuristics.
  // - DFS visits HLOs in postorder, with a heuristic to decide the order of
  //   children.
  // - Postorder does not use any heuristics.
  // List wins for most of our benchmarks; postorder-based schedulers win for
  // some RNNs.
  int64_t list_memory;
  TF_ASSIGN_OR_RETURN(
      HloSchedule list_sequence,
      ComputationSchedulerToModuleScheduler(ListMemoryScheduler, {})(
          module, points_to_analysis, alias_analysis, size_function,
          execution_threads, &list_memory));

  VLOG(2) << "Min-memory list sequence: " << HumanReadableNumBytes(list_memory);

  int64_t dfs_memory;
  TF_ASSIGN_OR_RETURN(
      HloSchedule dfs_sequence,
      ComputationSchedulerToModuleScheduler(DFSMemoryScheduler, {})(
          module, points_to_analysis, alias_analysis, size_function,
          execution_threads, &dfs_memory));
  VLOG(2) << "Min-memory dfs sequence: " << HumanReadableNumBytes(dfs_memory);

  int64_t post_order_memory;
  TF_ASSIGN_OR_RETURN(
      HloSchedule post_order_sequence,
      ComputationSchedulerToModuleScheduler(PostOrderMemoryScheduler, {})(
          module, points_to_analysis, alias_analysis, size_function,
          execution_threads, &post_order_memory));
  VLOG(2) << "Min-memory post order sequence: "
          << HumanReadableNumBytes(post_order_memory);

  auto min_memory = std::min({dfs_memory, post_order_memory, list_memory});
  if (peak_memory) {
    *peak_memory = min_memory;
  }

  if (min_memory == list_memory) {
    VLOG(2) << "Chose min-memory list sequence: "
            << HumanReadableNumBytes(list_memory);
    return list_sequence;
  } else if (min_memory == dfs_memory) {
    VLOG(2) << "Chose min-memory dfs sequence: "
            << HumanReadableNumBytes(dfs_memory);
    return dfs_sequence;
  } else {
    VLOG(2) << "Chose min-memory post_order sequence: "
            << HumanReadableNumBytes(post_order_memory);
    return post_order_sequence;
  }
}

StatusOr<HloSchedule> ScheduleModule(
    const HloModule* module, const BufferValue::SizeFunction& size_function,
    const ModuleSchedulerAlgorithm& algorithm,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    int64_t* peak_memory) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<TuplePointsToAnalysis> points_to_analysis,
                      TuplePointsToAnalysis::Run(module));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(module));

  TF_ASSIGN_OR_RETURN(HloSchedule schedule,
                      (algorithm ? algorithm : DefaultModuleScheduler)(
                          module, *points_to_analysis, *alias_analysis,
                          size_function, execution_threads, peak_memory));

  TF_RETURN_IF_ERROR(schedule.Verify());

  return std::move(schedule);
}

StatusOr<HloInstructionSequence> ScheduleComputation(
    HloComputation* computation, const BufferValue::SizeFunction& size_function,
    const MemorySchedulerPostprocessor& postprocessor) {
  CHECK(!computation->IsFusionComputation());
  TF_ASSIGN_OR_RETURN(std::unique_ptr<TuplePointsToAnalysis> points_to_analysis,
                      TuplePointsToAnalysis::Run(computation->parent()));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloAliasAnalysis> alias_analysis,
                      HloAliasAnalysis::Run(computation->parent()));
  absl::flat_hash_map<const HloComputation*, int64_t> empty_map;
  return ScheduleComputationHelper(
      computation, *points_to_analysis, *alias_analysis, size_function,
      /*algorithm=*/nullptr, empty_map, postprocessor,
      /*peak_memory=*/nullptr);
}

HloMemoryScheduler::HloMemoryScheduler(
    const BufferValue::SizeFunction& size_function,
    const ModuleSchedulerAlgorithm& algorithm)
    : size_function_(size_function), algorithm_(algorithm) {}

StatusOr<bool> HloMemoryScheduler::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(
      HloSchedule schedule,
      ScheduleModule(module, size_function_, algorithm_, execution_threads));
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));
  return true;
}

StatusOr<bool> HloTrivialScheduler::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloSchedule schedule(module);
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    if (!computation->IsFusionComputation()) {
      HloInstructionSequence& computation_sequence =
          schedule.GetOrCreateSequence(computation);
      FunctionVisitor visitor(
          [&computation_sequence](HloInstruction* instruction) {
            computation_sequence.push_back(instruction);
            return OkStatus();
          });
      visitor.ReserveVisitStates(computation->instruction_count());
      TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    }
  }
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));
  return true;
}

StatusOr<bool> HloDescheduler::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = module->has_schedule();
  module->clear_schedule();
  return changed;
}

}  // namespace xla
