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
            intervals.add(intervaltree.Interval(lb, ub))
        intervals.merge_overlap()
        
        ts = []
        for interval in intervals:
            ts += list(range(interval.begin, interval.end))
        
        ts = sorted(ts)
        self.iter = iter(ts[start_offset:])

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
                 max_memory_usage: {max_memory_usage}.")     
    scheduler_opti = LpProblem("scheduler optimization", LpMinimize)
    
    

    # Create Varaibles.
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
    
    for id in range(num_logical_buffer):
        live_range = max_useful_liverange[id]
        prev = TimeStepsForEdges(live_range)
        curr = TimeStepsForEdges(live_range, start_offset=1)
        
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
        asap_src, alap_src = max_useful_liverange[id][0][0], max_useful_liverange[id][0][1]
        scheduler_opti += preserve_vars[id][lb] == 0, \
                          "preserve_var_0_at_" + str(t) + "_buffer_ " + str(id)
        
        for t in TimeStepsForEdges(live_range):
            if t > alap_src:
                scheduler_opti += generate_vars[id][t] == 0, \
                                  "generate_var_less_alap_src_buffer_" + str(id)
        
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

        for t in range(asap_src, alap_src + 1):
            # Add siblings generated at the same time constraints.
            siblings = dependencies[id][3]
            for sib in siblings:
                scheduler_opti += generate_vars[id][t] == \
                                  generate_vars[sib][t], \
                                  f"generate_var_buffer_{id}_sib_{sib}"

            # Add precedencies generated later than successors constraints.
            precedencies = dependencies[id][0]
            for pre in precedencies:
                scheduler_opti += generate_vars[id][t] <= preserve_vars[pre][t], \
                                  f"generate_var_{id}_later_preserve_var_{pre}"
                
    # Force the generation of each buffer exactly once.
    for id, ts in generate_vars:
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


    # Memory usage at each timestep
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
    v = LpVariable(
        "peak_memory_usage", 
        lowBound=min_peak_memory // gcd,
        upper_bound=max_memory_usage,
    )
    
    for ts, mem in mem_at_timestep.items():
        scheduler_opti += v >= mem, f"peak_memory_greater_than_any_time_{ts}"

    msg = False
    time_limit = 600
    solver = pulp.PILP_CBC_CMD(mip=True,
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

    return created_time
