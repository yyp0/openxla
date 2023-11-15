import logging
import math
import time
import sys
from collections import defaultdict, OrderedDict

import multiprocessing
import intervaltree
import pulp
from pulp import LpVariable, LpProblem, LpMinimize, lpSum, lpDot, LpStatus


class TimeStepsForEdges:
    def __init__(self, live_range, start_offset=0): 
        intervals = intervaltree.IntervalTree()
        for (lb, ub) in live_range:
            if lb == ub:
                ub += 1
            intervals.add(intervaltree.Interval(lb, ub))
        intervals.merge_overlaps()
        
        ts = []
        for interval in intervals:
            ts += list(range(interval.begin, interval.end))
        
        ts = sorted(ts)
        self.iter = iter(ts[start_offset:])

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.iter.__next__()

class TimeStepsForMultiBuffers:
    def __init__(self, live_ranges):
        max_lb, min_ub = 0, sys.maxsize
        for live_range in live_ranges:
            lb, ub = ComputeSpan(live_range)
            max_lb = max(max_lb, lb)
            min_ub = min(min_ub, ub)
        
        ts = list(range(max_lb, min_ub))
        ts = sorted(ts)
        self.iter = iter(ts)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.iter.__next__()


class DensePreserveVarsMap:
    def __init__(self, sparse_map):
        local_map = {}
        maxi = 0
        mini = math.inf
        for i in sparse_map:
            if i > maxi:
                maxi = i
            if i < mini:
                mini = i
        for i in range(maxi, mini - 1, -1):
            assert i >= mini
            assert i <= maxi
            if i not in sparse_map:
                local_map[i] = local_map[i + 1]
            else:
                local_map[i] = sparse_map[i]
        self.local_map = OrderedDict(sorted(local_map.items()))

    def __getitem__(self, index):
        return self.local_map[index]

    def items(self):
        return self.local_map.items()


def ComputeSpan(liverange):
    lb = liverange[0][0]
    ub = 0
    for span in liverange[1:]:
        ub = max(ub, span[1])

    return (lb, ub)


def call_solver_serialized(max_useful_liverange,
                           dependencies,
                           logical_buffer_size,  
                           no_succ_buffer,
                           min_peak_memory,    
                           max_memory_usage,
                           persistent_buffer,
                           buffer_alias,):
    print("Start to launch solver.")

    print(f"max_useful_liverange: {max_useful_liverange} \n \
                 dependencies: {dependencies} \n \
                 logical_buffer_size: {logical_buffer_size} \n \
                 no_succ_buffer: {no_succ_buffer} \n \
                 min_peak_memory: {min_peak_memory} \n \
                 max_memory_usage: {max_memory_usage} \n \
                 persistent_buffer: {persistent_buffer} \n \
                 buffer_alias: {buffer_alias}.")     
    scheduler_opti = LpProblem("scheduler optimization", LpMinimize)
    
    # import pdb;pdb.set_trace()

    # Create Variables.
    print(f"Create to create variables.")
    num_logical_buffer = len(max_useful_liverange)   
    generate_vars = defaultdict(lambda: {})
    preserve_vars = defaultdict(lambda: {})
     
    for id in range(num_logical_buffer):
        live_range = max_useful_liverange[id]
        for ts in TimeStepsForEdges(live_range):
            v = LpVariable('generate_variable_for_'+str(id)+"_at_"+str(ts), 
                           0, 1, pulp.LpBinary)
            generate_vars[id][ts] = v
            v = LpVariable('preserve_variable_for_'+str(id)+"_at_"+str(ts),
                           0, 1, pulp.LpBinary)
            preserve_vars[id][ts] = v    
    
    print(f"Start to buil constraints.")
    for id in range(num_logical_buffer):
        live_range = max_useful_liverange[id]
        prev = TimeStepsForEdges(live_range)
        curr = TimeStepsForEdges(live_range, start_offset=1)
        
        print(f"Build constraints for correctness of C and P.")
        # Correctness constraints for created variables.
        for t in curr:
            p = prev.__next__()

            scheduler_opti += preserve_vars[id][t] <= preserve_vars[id][p] \
                                                    + generate_vars[id][p], \
                              "precedence_at_" + str(t) + "_buffer_ " + str(id)
            scheduler_opti += preserve_vars[id][t] + generate_vars[id][t] \
                              <= 1, \
                              "at_most_one_at_" + str(t) + "_buffer_ " + str(id)
        
        # Simplify the problem.
        print("Build constraints to simplify the problem.")
        asap_src, alap_src = max_useful_liverange[id][0][0], max_useful_liverange[id][0][1]
        scheduler_opti += preserve_vars[id][asap_src] == 0, \
                          "preserve_var_0_at_" + str(t) + "_buffer_ " + str(id)
        
        for t in TimeStepsForEdges(live_range):
            if t > alap_src:
                scheduler_opti += generate_vars[id][t] == 0, \
                                  f"generate_var_less_alap_src_buffer_{id}_at_{t}"
        
        if logical_buffer_size[id] == 0:
            prev = TimeStepsForEdges(live_range)
            for t in TimeStepsForEdges(live_range, start_offset=1):
                if t <= alap_src:
                    scheduler_opti += preserve_vars[id][t] \
                                      >= preserve_vars[id][p] + generate_vars[id][p], \
                                      "ctrl_edge_preserve_var_buffer_" + str(id)
                else:
                    scheduler_opti += preserve_vars[id][t] == 1, \
                                      "ctrl_edge_all_ready_buffer_" + str(id)

        # Add siblings generated at the same time constraints.
        print("Build constraints to protect the correctness of siblings and precedencies.")
        siblings = dependencies[id][2]
        live_ranges = []
        for sib in siblings:
            live_ranges.append(max_useful_liverange[sib])
        
        # import pdb;pdb.set_trace()
        for sib in siblings:
            for t in range(asap_src, alap_src + 1):
                scheduler_opti += generate_vars[id][t] == \
                                generate_vars[sib][t], \
                                f"generate_var_buffer_{id}_sib_{sib}_at_{t}"

        # Add precedencies generated before successors constraints.
        precedencies = dependencies[id][0]
        for pre in precedencies:
            live_ranges = [max_useful_liverange[id], max_useful_liverange[pre]]
            for t in TimeStepsForMultiBuffers(live_ranges):
                scheduler_opti += generate_vars[id][t] <= preserve_vars[pre][t], \
                                f"generate_var_{id}_later_precedencies_{pre}_at_{t}"

    print("Force the generation of each buffer.")       
    # Force the generation of each buffer exactly once.
    for id, ts in generate_vars.items():
        s = 0
        for v in ts.values():
            s += v

        scheduler_opti += s == 1, f"force_{id}_generated_once"

    for buffer_set in no_succ_buffer:   # Get no_succ_buffer from hlo graph.
        if len(buffer_set) <= 2:
            continue
        
        alap_instr = buffer_set[-1]
        timesteps = set(TimeStepsForEdges(max_useful_liverange[buffer_set[0]]))
        for i in range(len(buffer_set)):
            if i == len(buffer_set) - 1:
                break   
            timesteps &= set(TimeStepsForEdges(max_useful_liverange[buffer]))
        timesteps = sorted(timesteps)

        sum_of_all_live = 0
        for t in timesteps:
            if t > alap_instr:
                break

            all_live = LpVariable(f'fanin_buffers_alive_at_ts_{t}')
            for buffer in buffer_set:
                scheduler_opti += all_live <= preserve_vars[buffer][t], \
                                  f"fanin_buffers_{buffer}_at_ts_{t}"
            sum_of_all_live += all_live
        scheduler_opti += sum_of_all_live >= 1, \
                          "force_fanin_buffers_live_at_same_time_one"

    # Constraints for persistent buffer and activations.
    print("Add constraints for persistent buffer and activations.")
    for id in range(num_logical_buffer):
        if False: # id in persistent_buffer:
            is_first = True
            for t in TimeStepsForEdges(max_useful_liverange[id]):
                if is_first:
                    scheduler_opti += generate_vars[id][t] == 1, \
                                      f"weight_{id}_generat_var_at_first_time_{t}"
                    scheduler_opti += preserve_vars[id][t] == 0, \
                                      f"weight_{id}_preserve_var_at_first_time_{t}"
                    is_first = False
                else:
                    scheduler_opti += preserve_vars[id][t] == 1, \
                                      f"weight_{id}_preserve_var_at_{t}"
            continue

        first = max_useful_liverange[id][0][1] + 1  # alap_src + 1
        last = 0
        for live in max_useful_liverange[id][1:]:
            last = max(last, live[0])   # max(asap[snk])
        for ts in TimeStepsForEdges(max_useful_liverange[id]):
            if ts < first or ts > last:
                continue
            scheduler_opti += preserve_vars[id][ts] == 1, \
                              f"buffer_{id}_must_be_preserved_util_last_used_at_{ts}"

    # Memory usage at each timestep
    print("Calculate memory usage at each timestep.")
    gcd = 4
    mem_at_timestep = defaultdict(lambda: 0)
    for id in range(num_logical_buffer):
        if logical_buffer_size[id] == 0:
            continue
        
        for t, v in DensePreserveVarsMap(preserve_vars[id]).items():
            mem_at_timestep[t] += v * (logical_buffer_size[id] // gcd)
        
        for t, v in generate_vars[id].items():
            mem_at_timestep[t] += v * (logical_buffer_size[id] // gcd)

    # Add objective.
    print("Add objective and related constraints.")
    v = LpVariable(
        "peak_memory_usage", 
        lowBound=min_peak_memory // gcd,
        upBound=max_memory_usage,
    )
    
    for ts, mem in mem_at_timestep.items():
        scheduler_opti += v >= mem, f"peak_memory_greater_than_any_time_{ts}"

    msg = False
    time_limit = 600
    # pulp.listSolvers(onlyAvailable=True)
    solver = pulp.PULP_CBC_CMD(mip=True,
                               msg=msg,
                               timeLimit=time_limit,
                               threads=multiprocessing.cpu_count())
    
    scheduler_opti.solve(solver)
    
    created_time = {}
    status = LpStatus[scheduler_opti.status]
    if status == "Optimal":
        # TODO(HuiyaoShu): validate the results. 
        for id, ts in generate_vars.items():
            for t in ts:
                if generate_vars[id][t].varValue >= 0.99:
                    created_time[id] = t
    else: 
        logging.error(f"Finish the solver, final status is {status}")

    for id, t in created_time.items():
        print(f"Created time for buffer {id}: {t}")

    return created_time

def main():
    max_useful_liverange = [[[0, 0], [1, 34], [1, 17]], [[0, 0], [2, 19], [13, 36]], [[0, 0], [1, 18]], [[0, 0], [1, 24]], [[0, 0], [5, 20], [11, 30]], [[0, 0], [5, 20], [11, 30]], [[0, 0], [5, 20], [11, 30]], [[0, 0], [9, 25], [11, 30]], [[0, 0], [9, 25], [11, 30]], [[0, 0], [9, 25], [11, 30]], [[2, 19], [5, 20]], [[11, 36], [27, 37]], [[11, 36], [27, 37]], [[13, 36], [27, 37]], [[5, 20], [6, 24], [6, 24], [6, 36], [6, 36]], [[5, 20], [9, 25], [6, 24], [6, 24], [6, 36], [6, 36], [27, 37]], [[5, 20], [9, 25], [6, 24], [6, 24], [6, 36], [6, 36], [27, 37]], [[5, 20], [6, 24], [6, 24], [6, 36], [6, 36], [27, 37]], [[5, 20], [6, 24], [6, 24], [6, 36], [6, 36], [27, 37]], [[9, 25], [10, 27], [10, 36]], [[9, 25], [11, 36], [11, 36], [11, 30], [10, 27], [10, 36]], [[9, 25], [10, 27], [10, 36], [27, 37]], [[2, 35], [3, 36], [27, 37]], [[1, 34], [2, 35]], [[2, 18], [3, 36], [5, 20], [9, 25], [27, 37]], [[1, 17], [2, 18]], [[11, 30], [12, 36], [12, 36], [12, 34], [12, 36]], [[11, 30], [12, 36], [12, 36], [12, 34], [12, 36], [27, 37]], [[11, 30], [12, 36], [12, 36], [12, 34], [12, 36], [27, 37]], [[11, 30], [13, 36], [13, 36], [12, 36], [12, 36], [12, 34], [12, 36], [27, 37]], [[11, 30], [12, 36], [12, 36], [12, 34], [12, 36], [27, 37]], [[27, 37], [37, 37]], [[1, 18], [2, 19], [11, 30]], [[1, 24], [9, 25]]] # [[[0, 0], [1, 3]], [[0, 0], [1, 3]], [[1, 3], [3, 3]]]
    dependencies = [[[], [23, 25], [0]], [[], [10, 13], [1]], [[], [32], [2]], [[], [33], [3]], [[], [14, 15, 16, 17, 18, 26, 27, 28, 29, 30], [4]], [[], [14, 15, 16, 17, 18, 26, 27, 28, 29, 30], [5]], [[], [14, 15, 16, 17, 18, 26, 27, 28, 29, 30], [6]], [[], [19, 20, 21, 26, 27, 28, 29, 30], [7]], [[], [19, 20, 21, 26, 27, 28, 29, 30], [8]], [[], [19, 20, 21, 26, 27, 28, 29, 30], [9]], [[1, 32], [14, 15, 16, 17, 18], [10]], [[20], [31], [11]], [[20], [31], [12]], [[1, 29], [31], [13]], [[24, 6, 5, 4, 10], [], [14, 15, 16, 17, 18]], [[24, 6, 5, 4, 10], [19, 20, 21, 31], [14, 15, 16, 17, 18]], [[24, 6, 5, 4, 10], [19, 20, 21, 31], [14, 15, 16, 17, 18]], [[24, 6, 5, 4, 10], [31], [14, 15, 16, 17, 18]], [[24, 6, 5, 4, 10], [31], [14, 15, 16, 17, 18]], [[7, 33, 16, 9, 24, 8, 15], [], [19, 20, 21]], [[7, 33, 16, 9, 24, 8, 15], [11, 12, 26, 27, 28, 29, 30], [19, 20, 21]], [[7, 33, 16, 9, 24, 8, 15], [31], [19, 20, 21]], [[23], [31], [22]], [[0], [22], [23]], [[25], [14, 15, 16, 17, 18, 19, 20, 21, 31], [24]], [[0], [24], [25]], [[8, 5, 4, 6, 32, 9, 20, 7], [], [26, 27, 28, 29, 30]], [[8, 5, 4, 6, 32, 9, 20, 7], [31], [26, 27, 28, 29, 30]], [[8, 5, 4, 6, 32, 9, 20, 7], [31], [26, 27, 28, 29, 30]], [[8, 5, 4, 6, 32, 9, 20, 7], [13, 31], [26, 27, 28, 29, 30]], [[8, 5, 4, 6, 32, 9, 20, 7], [31], [26, 27, 28, 29, 30]], [[15, 30, 13, 29, 28, 21, 12, 24, 22, 11, 18, 17, 16, 27], [], []], [[2], [10, 26, 27, 28, 29, 30], [32]], [[3], [19, 20, 21], [33]]]  # [[[], [2], [0]], [[], [2], [1]], [[0, 1], [], []]]
    logical_buffer_size = [8, 4, 524288, 2097152, 4, 8, 4, 8, 4, 4, 2048, 524288, 524288, 4, 32, 2097152, 2097152, 2097152, 2097152, 16, 2048, 2097152, 1024, 1024, 4096, 4096, 32, 524288, 524288, 1024, 524288, 112, 524288, 2097152]  # [4, 4, 4]
    no_succ_buffer = [] 
    min_peak_memory = 13113460 # 12
    max_memory_usage = 18365680 # 12
    persistent_buffer = []
    buffer_alias = []

    call_solver_serialized(max_useful_liverange,
                           dependencies,
                           logical_buffer_size,
                           no_succ_buffer,
                           min_peak_memory,
                           max_memory_usage,
                           persistent_buffer,
                           buffer_alias)
    
main()