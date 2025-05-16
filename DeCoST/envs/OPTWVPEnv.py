from dataclasses import dataclass
import torch
import os, pickle
import sys
import numpy as np
from collections import namedtuple
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from TOPTWVP_Gurobi_Solver import TOPTWVP_Gurobi_Solver
import time

__all__ = ['OPTWVPEnv']
dg = sys.modules[__name__]

OPTWVP_SET = namedtuple("OPTWVP_SET",
                       ["node_loc",  # Node locations 1
                        "node_tw",  # node time windows 5
                        "durations",  # service duration per node 6
                        "service_window",  # maximum of time units 7
                        "time_factor", "loc_factor"]) # 归一化参数

@dataclass
class Reset_State:
    node_xy: torch.Tensor = None
    max_service_time: torch.Tensor = None
    node_tw_start: torch.Tensor = None
    node_tw_end: torch.Tensor = None
    profit: torch.Tensor = None 
    node_s_tw_start: torch.Tensor = None
    node_s_tw_end: torch.Tensor = None
    distance: torch.Tensor = None

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    START_NODE: torch.Tensor = None
    PROBLEM: str = None
    selected_count: int = None
    current_node: torch.Tensor = None
    ninf_mask: torch.Tensor = None
    finished: torch.Tensor = None
    infeasible: torch.Tensor = None
    current_time: torch.Tensor = None
    length: torch.Tensor = None
    current_coord: torch.Tensor = None
    profit: torch.Tensor = None

class OPTWVPEnv:
    def __init__(self, **env_params):

        self.problem = "OPTWVP"
        self.env_params = env_params
        self.hardness = env_params['hardness']
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size'] # pomo采样大小
        self.max_tw_size = env_params['max_tw_size'] if 'max_tw_size' in env_params.keys() else 100
        self.loc_scaler = env_params['loc_scaler'] if 'loc_scaler' in env_params.keys() else None
        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in env_params.keys() else env_params['device']

        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.START_NODE = None
        self.node_xy = None
        self.node_service_time = None
        self.max_service_time = None
        self.node_tw_start = None
        self.node_tw_end = None
        self.node_s_tw_start = None
        self.node_s_tw_end = None
        self.speed = 1.0
        self.max_travel_distance = None
        self.total_profit = None
        self.profit = None
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.timestamps = None
        self.infeasibility_list = None
        self.timeout_list = None
        self.step_lengths = None
        self.log_travel_time = None
        self.visited_ninf_flag = None
        self.simulated_ninf_flag = None
        self.global_mask = None
        self.global_mask_ninf_flag = None
        self.out_of_tw_ninf_flag = None
        self.ninf_mask = None
        self.finished = None
        self.infeasible = None
        self.current_time = None
        self.length = None
        self.current_coord = None
        self.reset_state = Reset_State()
        self.step_state = Step_State()
        self.stage = 5
        # 1 POMO method;  5 batch method

    def load_problems(self, batch_size, problems=None, aug_factor=1, normalize=True):
        # 处理输入问题数据
        if problems is not None:
            node_xy, service_time, tw_start, tw_end, profit, s_tw_start, s_tw_end, max_travel_distance = problems
        else:
            node_xy, service_time, tw_start, tw_end, profit, s_tw_start, s_tw_end, max_travel_distance = self.get_random_problems(batch_size, self.problem_size, max_tw_size=self.max_tw_size)

        # 归一化数据
        if normalize:
            # Normalize as in DPDP (Kool et. al)
            loc_factor = 100
            node_xy = node_xy / loc_factor  # Normalize
            tw_start = tw_start / loc_factor
            tw_end = tw_end / loc_factor
            s_tw_start = s_tw_start / loc_factor
            s_tw_end = s_tw_end / loc_factor
            tw_end[:, 0] = (torch.cdist(node_xy[:, None, 0], node_xy[:, 1:]).squeeze(1) + tw_end[:, 1:]).max(dim=-1)[0]
        self.batch_size = node_xy.size(0)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                node_xy = self.augment_xy_data_by_8_fold(node_xy)
                service_time = service_time.repeat(8, 1)
                tw_start = tw_start.repeat(8, 1)
                tw_end = tw_end.repeat(8, 1)
                profit = profit.repeat(8,1)
                s_tw_start = s_tw_start.repeat(8, 1)
                s_tw_end = s_tw_end.repeat(8, 1)
                max_travel_distance = max_travel_distance.repeat(8)
            else:
                raise NotImplementedError
        self.node_xy = node_xy
        self.max_service_time = service_time
        self.node_tw_start = tw_start
        self.node_tw_end = tw_end
        self.node_tw_end[:,0] = max_travel_distance[0].item()
        self.node_s_tw_start = s_tw_start
        self.node_s_tw_end = s_tw_end
        self.max_service_time = self.node_tw_end - self.node_tw_start
        self.max_service_time[:,0] = 0
        self.profit = profit
        self.max_travel_distance = max_travel_distance[:,None].expand(-1, self.pomo_size)
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)

        self.reset_state.node_xy = node_xy
        self.reset_state.max_service_time = service_time
        self.reset_state.node_tw_start = tw_start
        self.reset_state.node_tw_end = tw_end
        self.reset_state.profit = profit
        self.reset_state.max_travel_distance = max_travel_distance[:,None].expand(-1, self.problem_size)
        self.reset_state.node_s_tw_start = s_tw_start
        self.reset_state.node_s_tw_end = s_tw_end
        x_diff = node_xy[:, :, 0].unsqueeze(2) - node_xy[:, :, 0].unsqueeze(1)
        y_diff = node_xy[:, :, 1].unsqueeze(2) - node_xy[:, :, 1].unsqueeze(1)
        self.distance = torch.sqrt(x_diff**2 + y_diff**2)
        self.reset_state.distance = self.distance

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.START_NODE = torch.arange(start=1, end=self.pomo_size + 1)[None, :].expand(self.batch_size, -1).to(self.device)
        self.step_state.PROBLEM = self.problem

        k_sparse = self.env_params["k_sparse"]
        node_xy_expanded = node_xy[:, :, None, :]  # (B, N, 1, 2)
        node_xy_expanded_T = node_xy[:, None, :, :]  # (B, 1, N, 2)
        distances = torch.sqrt(torch.sum((node_xy_expanded - node_xy_expanded_T) ** 2, dim=-1))  # (B, N, N)
        diag_mask = torch.eye(self.problem_size).unsqueeze(0).repeat(self.batch_size,1,1) * (1e9)
        distances += diag_mask
        if k_sparse < self.problem_size:
            self.is_sparse = True
            print("Sparse, ", k_sparse)
            _, topk_indices1 = torch.topk(distances, k=k_sparse, dim=-1, largest=False)
            dist_neighbors_index = torch.cat([
                torch.repeat_interleave(torch.arange(self.problem_size), repeats=k_sparse).reshape(1, self.problem_size,-1).repeat(self.batch_size,1,1).unsqueeze(-1),
                topk_indices1.unsqueeze(-1)
            ], dim=-1)

            start_node_tw_start = tw_start[:, :1]
            tw_start_differences = tw_start - start_node_tw_start
            tw_start_differences[tw_start_differences <= 0] = float('inf')
            _, topk_indices2 = torch.topk(tw_start_differences, k=k_sparse, dim=-1, largest=False)
            edge_index0 = torch.cat([
                torch.repeat_interleave(torch.tensor(0), repeats=k_sparse).reshape(1,-1).repeat(self.batch_size,1).unsqueeze(-1),
                topk_indices2.unsqueeze(-1)
            ], dim=-1)

            start_times = tw_start[:,1:].unsqueeze(-1).expand(-1, -1, self.problem_size - 1)
            end_times = tw_end[:,1:].unsqueeze(-1).expand(-1, -1, self.problem_size - 1)
            start_max = torch.max(start_times, start_times.transpose(1,2))
            end_min = torch.min(end_times, end_times.transpose(1, 2))
            overlap_matrix = torch.clamp(end_min - start_max, min=0)
            eye_matrix = torch.eye(self.problem_size-1).unsqueeze(0).repeat(self.batch_size, 1, 1).bool()
            overlap_matrix[eye_matrix] = 0.  # ignore self
            del eye_matrix
            _, topk_indices3 = torch.topk(overlap_matrix, k=k_sparse, dim=-1)
            topk_indices3 += 1  # since we remove the first node (start node) in overlap_matrix
            edge_index1 = torch.cat([
                torch.repeat_interleave(torch.arange(1, self.problem_size), repeats=k_sparse).reshape(1, self.problem_size-1,-1).repeat(self.batch_size,1,1).unsqueeze(-1),
                topk_indices3.unsqueeze(-1)
            ], dim=-1)
            tw_neighbors_index = torch.concat([edge_index0.unsqueeze(1), edge_index1], dim=1)
            self.neighbour_index = torch.concat([dist_neighbors_index, tw_neighbors_index], dim=2)
            
            self.k_neigh_ninf_flag =torch.full((self.batch_size, self.problem_size, self.problem_size), float('-inf'))
            indices = self.neighbour_index.view(self.batch_size, -1, 2)
            self.k_neigh_ninf_flag[torch.arange(self.batch_size).view(-1, 1).expand_as(indices[:, :, 0]), indices[:, :, 0], indices[:, :, 1]] = 0
            self.k_neigh_ninf_flag[torch.arange(self.batch_size).view(-1, 1).expand_as(indices[:, :, 0]), indices[:, :, 1], indices[:, :, 0]] = 0
        else:
            self.is_sparse = False

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)
        self.timestamps = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)
        self.log_travel_time = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)

        self.timeout_list = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)
        self.infeasibility_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.bool).to(self.device) # True for causing infeasibility
        self.step_lengths = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)

        self.node_service_time = torch.zeros((self.batch_size, self.pomo_size, 0)).to(self.device)

        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.simulated_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.global_mask_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.global_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.out_of_tw_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size)).to(self.device)

        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        self.infeasible = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)

        self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        self.length = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        self.current_coord = self.node_xy[:, :1, :]  # depot

        self.total_profit = 0
        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.infeasible = self.infeasible
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected, service_time_normed, visit_mask_only =True, out_reward = False):
        self.selected_count += 1
        self.current_node = selected
        current_coord = self.node_xy[torch.arange(self.batch_size)[:, None], selected]
        new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)
        self.log_travel_time = torch.cat((self.log_travel_time, new_length[:, :, None]), dim=2)
        self.length = self.length + new_length
        self.current_coord = current_coord

        self.step_lengths = torch.cat((self.step_lengths, new_length[:, :, None]), dim=2)  # 按 `step` 维度拼接
        return_to_depot_length = (self.node_xy[torch.arange(self.batch_size)[:, None], selected] - 
                        self.node_xy[:, None, 0, :]).norm(p=2, dim=-1)          
        service_time = service_time_normed * torch.min(self.max_service_time[torch.arange(self.batch_size)[:, None], selected], self.max_travel_distance - new_length - return_to_depot_length - self.current_time)

        if self.node_service_time.shape[2] != 0:
            self.current_time = (torch.max(self.current_time + new_length / self.speed + self.node_service_time[:,:,-1],
                                       self.node_tw_start[torch.arange(self.batch_size)[:, None], selected]) + self.node_service_time[:,:,-1])
        else:
            self.current_time = (torch.max(self.current_time + new_length / self.speed,
                                    self.node_tw_start[torch.arange(self.batch_size)[:, None], selected]))
        self.timestamps = torch.cat((self.timestamps, self.current_time[:, :, None]), dim=2)
        self.node_service_time = torch.cat((self.node_service_time, service_time[:, :, None]), dim=2)
        newly_finished1 = (self.current_time + service_time > self.max_travel_distance - return_to_depot_length)
        to_unvisited_distance = (self.node_xy[:, None, :, :] - self.node_xy[self.BATCH_IDX, self.current_node][:, :, None, :]).norm(p=2, dim=-1)
        next_unvisited_time = (torch.max(self.current_time[:,:,None] + service_time[:,:,None] + to_unvisited_distance / self.speed, self.node_tw_start[:,None,:]))
        to_depot_distance = (self.node_xy - self.node_xy[:, 0, :].unsqueeze(1)).norm(p=2, dim=-1)  # (batch, problem_size)
        newly_finished2 = (next_unvisited_time + to_depot_distance[:, None, :] > self.max_travel_distance[:, :, None]).all(dim=2) 
        if self.selected_node_list.shape[2] != 0:
            newly_finished3 = (selected==0)
        else:
            newly_finished3 = torch.zeros_like(selected, dtype=torch.bool)
        round_error_epsilon = 0.00001
        next_arrival_time = torch.max(self.current_time[:, :, None] + service_time[:,:,None] + (self.current_coord[:, :, None, :] - self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) / self.speed,
                                 self.node_tw_start[:, None, :].expand(-1, self.pomo_size, -1))
        out_of_tw = next_arrival_time > self.node_tw_end[:, None, :].expand(-1, self.pomo_size, -1) + round_error_epsilon
        self.out_of_tw_ninf_flag[out_of_tw] = float('-inf')

        self.finished = self.finished | newly_finished3 # | self.infeasible

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        unvisited_mask = (self.visited_ninf_flag == 0)  
        exceed_distance_mask = (next_unvisited_time + to_depot_distance[:, None, :] > self.max_travel_distance[:, :, None])
        exceed_tw_mask = (self.current_time[:, :, None] > self.node_tw_end[:, None, :].expand(-1, self.pomo_size, -1))
        self.visited_ninf_flag[unvisited_mask & out_of_tw] = float('-inf')
        self.visited_ninf_flag[unvisited_mask & exceed_distance_mask] = float('-inf')
        self.visited_ninf_flag[:,:,0] = 0

        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        self.ninf_mask = self.visited_ninf_flag.clone()
        if not visit_mask_only:
            self.ninf_mask[out_of_tw] = float('-inf')
        finished_mask = self.finished.unsqueeze(-1).expand(-1, -1, self.problem_size)  # shape: (batch, pomo, problem_size)
        self.ninf_mask[finished_mask] = float('-inf')
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.infeasible = self.infeasible
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        done = self.finished.all()
        if done:
            if not out_reward:
                opt_solution, total_timeout_reward, timeout_nodes_reward, reward, service_time_sl_reward = self._get_total_profit()
                infeasible = (reward < 0) # placeholder since the results is for sure feasible
            else:
                opt_solution, total_timeout_reward, timeout_nodes_reward, opt_reward, service_time_sl_reward = self._get_total_profit()
                reward = [opt_reward, total_timeout_reward, timeout_nodes_reward, service_time_sl_reward]
                infeasible = (opt_reward[0] < 0) # placeholder since the results is for sure feasible
        else:
            reward = None
            infeasible = 0.
        return self.step_state, reward, done, infeasible

    def _get_total_profit(self):
        selected_profit_pertime = self.profit[:, None, :].expand(-1, self.pomo_size, -1).gather(dim=2, index=self.selected_node_list)
        optimal_s = self.node_service_time.clone()
        selected_nodes = self.selected_node_list
        selected_tw_start = self.node_tw_start[:, None, :].expand(-1, self.pomo_size, -1).gather(dim=2, index=self.selected_node_list)
        selected_tw_end = self.node_tw_end[:, None, :].expand(-1, self.pomo_size, -1).gather(dim=2, index=self.selected_node_list)
        
        if self.stage == 1:
            start_time = time.time()
            batch_size, pomo_size, num_selected = selected_nodes.shape
            optimal_reward = (selected_profit_pertime * optimal_s).sum(dim=2)
        elif self.stage == 5:
            timestamps = self.timestamps
            timeout = torch.where((timestamps - selected_tw_end) > 0, timestamps - selected_tw_end, torch.tensor(0.0, device=timestamps.device))

            batch_size, pomo_size, num_selected = selected_nodes.shape
            optimal_reward = torch.zeros((batch_size, pomo_size))  # 新增存储 reward
            optimal_reward2 = torch.zeros((batch_size, pomo_size))  # 新增存储 reward
            total_timeout_reward = torch.zeros((batch_size, pomo_size))
            timeout_nodes_reward = torch.zeros((batch_size, pomo_size))
            num_routes = 1 

            n_nodes = self.profit.shape[1] + 1
            start_time = time.time()
            d_batch = torch.zeros((batch_size, pomo_size, num_selected), device=selected_nodes.device)
            s_batch = torch.zeros((batch_size, pomo_size, num_selected), device=selected_nodes.device)
            
            d_batch[:, :, 0] = 0.0
            s_batch[:, :, 0] = selected_tw_start[:, :, 0]
            s_batch[:, :, 1] = torch.maximum(selected_tw_start[:, :, 1], self.log_travel_time[:, :, 1])
            tw_size = selected_tw_end - selected_tw_start
            for i in range(1, num_selected-1):
                next_tw_constraint = selected_tw_start[:, :, i+1] - s_batch[:, :, i] - self.log_travel_time[:, :, i+1]
                s_batch[:, :, i+1] = torch.maximum(s_batch[:, :, i] + self.log_travel_time[:, :, i+1], selected_tw_start[:, :, i+1])

            sorted_indices = torch.argsort(selected_profit_pertime[: ,: ,:-1], dim=2, descending=True) 
            batch_indices = torch.arange(batch_size, device=sorted_indices.device)[:, None].expand(batch_size, pomo_size)
            pomo_indices = torch.arange(pomo_size, device=sorted_indices.device)[None, :].expand(batch_size, pomo_size)
            for i in range(0, num_selected):
                budget_batch = self.max_travel_distance[:, None, 0] - s_batch[:, :,-1]
                cur_idx = sorted_indices[:, :, i]  # 获取当前排序位置对应的节点索引
                end_mask = (selected_nodes[batch_indices,pomo_indices,cur_idx] == 0) # | (budget_batch <= 0) # ends of route
                if end_mask.all():
                    break
                non_calculate_mask = end_mask | (d_batch[batch_indices,pomo_indices,cur_idx] == tw_size[batch_indices,pomo_indices,cur_idx])
                if non_calculate_mask.all(): # no room for optimal
                    continue
                values = torch.stack([
                            tw_size[batch_indices,pomo_indices,cur_idx],
                            selected_tw_end[batch_indices,pomo_indices,cur_idx+1] 
                            - self.log_travel_time[batch_indices,pomo_indices,cur_idx+1] 
                            - s_batch[batch_indices,pomo_indices,cur_idx]
                        ], dim=2)
                travel_time_cumsum = torch.cumsum(self.log_travel_time, dim=2)
                d_batch_cumsum = torch.cumsum(d_batch, dim=2)
                max_k = (num_selected - cur_idx - 1).max().item()
                for k in range(1, max_k):
                    idx_k1 = torch.clamp(cur_idx + k + 1, max=num_selected-1)
                    idx_k = torch.clamp(cur_idx + k, max=num_selected-1)
                    travel_time_sum = travel_time_cumsum[batch_indices,pomo_indices,idx_k1] - travel_time_cumsum[batch_indices,pomo_indices,cur_idx]
                    d_sum = d_batch_cumsum[batch_indices,pomo_indices,idx_k] - d_batch_cumsum[batch_indices,pomo_indices,cur_idx]
                    values = torch.cat([values, 
                                        (selected_tw_end[batch_indices,pomo_indices,idx_k1] - 
                                         travel_time_sum - s_batch[batch_indices,pomo_indices,cur_idx] - d_sum
                                         ).unsqueeze(2)
                                        ], dim=2)
                d_batch[batch_indices,pomo_indices,cur_idx] = torch.where(non_calculate_mask, 
                                                                          d_batch[batch_indices,pomo_indices,cur_idx], 
                                                                          torch.min(values, dim=2).values)
                # print('cnm')
                # iteratively update S 
                # flush S in from start
                for j in range(1, num_selected):
                    s_batch[:,:,j] = torch.maximum(
                        s_batch[:,:,j-1] + d_batch[:,:,j-1] + self.log_travel_time[:,:,j],
                        selected_tw_start[:,:,j]
                        )
            optimal_s = d_batch
            optimal_reward = (selected_profit_pertime * optimal_s).sum(dim=2)
            # ==============================================================================
            
        arrival_time = torch.zeros_like(optimal_s)
        arrival_time[:,:,0] = optimal_s[:,:,0]
        for i in range(1, optimal_s.shape[2]):
            arrival_time[:,:,i] = torch.maximum(
                arrival_time[:,:,i-1] + optimal_s[:,:,i-1] + self.log_travel_time[:,:,i], 
                selected_tw_start[:,:,i]
                )
        timeout = torch.where((arrival_time - selected_tw_end) > 0, arrival_time - selected_tw_end, torch.tensor(0.0, device=arrival_time.device))
        total_timeout_reward = - (timeout).sum(dim=2)
        timeout_nodes_reward = - torch.where(timeout > 0, torch.ones_like(timeout), timeout).sum(dim=2).int()

        service_time_sl_reward = ((selected_profit_pertime * (self.node_service_time - optimal_s)).sum(dim=2) / (1e-5 + self.log_travel_time.sum(dim=2))) ** 2 #  + (optimal_reward / self.log_travel_time.sum(dim=2)) ** 2
        end_time = time.time()
        total_time = end_time - start_time

        return optimal_s, 0 * total_timeout_reward, 0 * timeout_nodes_reward, 1000 * optimal_reward, 1000 * service_time_sl_reward

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        if self.loc_scaler:
            segment_lengths = torch.round(segment_lengths * self.loc_scaler) / self.loc_scaler

        travel_distances = segment_lengths.sum(2)
        return travel_distances

    def get_gurobi_optimal_solution(self):
        solver = TOPTWVP_Gurobi_Solver(
            self.max_travel_distance, 
            self.node_xy, 
            self.max_service_time, 
            self.node_tw_start, 
            self.node_tw_end, 
            self.node_s_tw_start, 
            self.node_s_tw_end,
            self.profit)
        print("start solving optimal solution ...")
        solution = solver.solve(num_routes=1)
        solver.get_solution_summary(solution)
        return solution

    def generate_dataset(self, num_samples, problem_size, path):
        data = self.get_random_problems(num_samples, problem_size, max_tw_size=self.max_tw_size)
        dataset = [attr.cpu().tolist() for attr in data]
        filedir = os.path.split(path)[0]
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
        with open(path, 'wb') as f:
            pickle.dump(list(zip(*dataset)), f, pickle.HIGHEST_PROTOCOL)
        print("Save OPTWVP dataset to {}".format(path))

    def load_dataset(self, path, offset=0, num_samples=10000, disable_print=True):
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset+num_samples]
            if not disable_print:
                print(">> Load {} data ({}) from {}".format(len(data), type(data), path))
        node_xy, service_time, tw_start, tw_end, profit, s_tw_start, s_tw_end, max_travel_distance = \
            [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data], \
            [i[4] for i in data], [i[5] for i in data], [i[6] for i in data], [i[7] for i in data]
        node_xy, service_time, tw_start, tw_end, profit, s_tw_start, s_tw_end, max_travel_distance = \
            torch.Tensor(node_xy), torch.Tensor(service_time), torch.Tensor(tw_start), torch.Tensor(tw_end), \
            torch.Tensor(profit), torch.Tensor(s_tw_start), torch.Tensor(s_tw_end), torch.Tensor(max_travel_distance)

        data = (node_xy, service_time, tw_start, tw_end, profit, s_tw_start, s_tw_end, max_travel_distance)
        return data

    def get_random_problems(self, batch_size, problem_size, coord_factor=100, max_tw_size=500):
        # generate random datasets 
        if self.hardness == "hard":
            # Taken from DPDP (Kool et. al)
            # Taken from https://github.com/qcappart/hybrid-cp-rl-solver/blob/master/src/problem/OPTWVP/environment/OPTWVP.py
            # max_tw_size = 1000 if tw_type == "da_silva" else 100
            # max_tw_size = problem_size * 2 if tw_type == "da_silva" else 100
            """
            :param problem_size: number of cities
            :param grid_size (=1): x-pos/y-pos of cities will be in the range [0, grid_size]
            :param max_tw_gap: maximum time windows gap allowed between the cities constituing the feasible tour
            :param max_tw_size: time windows of cities will be in the range [0, max_tw_size]
            :return: a feasible OPTWVP instance randomly generated using the parameters
            """
            node_xy = torch.rand(size=(batch_size, problem_size, 2)) * coord_factor  # (batch, problem, 2)
            travel_time = torch.cdist(node_xy, node_xy, p=2, compute_mode='donot_use_mm_for_euclid_dist') / self.speed # (batch, problem, problem)

            random_solution = torch.arange(1, problem_size).repeat(batch_size, 1)
            for i in range(batch_size):
                random_solution[i] = random_solution[i][torch.randperm(random_solution.size(1))]
            zeros = torch.zeros(size=(batch_size, 1)).long()
            random_solution = torch.cat([zeros, random_solution], dim=1)
            
            if problem_size == 50:
                max_budget = 10 # 4 + 100/400 * 50/2 
            if problem_size == 100:
                max_budget = 4 + 100 / 400 * problem_size / 2
            elif problem_size == 500:
                max_budget = 50

            time_windows = torch.zeros((batch_size, problem_size, 2))
            time_windows[:, 0, :] = torch.tensor([0, max_budget * coord_factor]).repeat(batch_size, 1)
            service_time_windows = time_windows # initalize

            total_dist = torch.zeros(batch_size)
            for i in range(1, problem_size):
                prev_city = random_solution[:, i - 1]
                cur_city = random_solution[:, i]

                cur_dist = travel_time[torch.arange(batch_size), prev_city, cur_city]
                total_dist += cur_dist

                # Style by Da Silva and Urrutia, 2010, "A VNS Heuristic for OPTWVP"
                rand_tw_lb = torch.rand(batch_size) * (max_tw_size / 2) + (total_dist - max_tw_size / 2)
                rand_tw_ub = torch.rand(batch_size) * (max_tw_size / 2) + total_dist

                # service time window use method of "the Tourist Trip Design Problem with TWVP using Incremental Local Search"
                service_tw_lb = rand_tw_lb
                service_tw_ub = rand_tw_lb + max_tw_size / 4

                time_windows[torch.arange(batch_size), cur_city, :] = torch.cat([rand_tw_lb.unsqueeze(1), rand_tw_ub.unsqueeze(1)], dim=1)
                service_time_windows[torch.arange(batch_size), cur_city, :] = torch.cat([service_tw_lb.unsqueeze(1), service_tw_ub.unsqueeze(1)], dim=1)

        else:
            raise NotImplementedError

        profit = torch.rand(size=(batch_size,problem_size)) * 10
        profit[:,0] = 0
        service_time = torch.zeros(size=(batch_size,problem_size))
        # Don't store travel time since it takes up much

        max_travel_distance = max_budget * torch.ones(size=(batch_size,))

        return node_xy, service_time, time_windows[:,:,0], time_windows[:,:,1], profit, service_time_windows[:,:,0], service_time_windows[:,:,1], max_travel_distance

    def augment_xy_data_by_8_fold(self, xy_data):
        x = xy_data[:, :, [0]]
        y = xy_data[:, :, [1]]
        dat1 = torch.cat((x, y), dim=2)
        dat2 = torch.cat((1 - x, y), dim=2)
        dat3 = torch.cat((x, 1 - y), dim=2)
        dat4 = torch.cat((1 - x, 1 - y), dim=2)
        dat5 = torch.cat((y, x), dim=2)
        dat6 = torch.cat((1 - y, x), dim=2)
        dat7 = torch.cat((y, 1 - x), dim=2)
        dat8 = torch.cat((1 - y, 1 - x), dim=2)
        aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
        return aug_xy_data

    def visualize_solution(self, batch_idx=0, pomo_idx=0, epoch=0):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle
        figsize=(12, 10)
        save_path = f"./sol_visualization/{epoch}"
        node_xy = self.node_xy[batch_idx].cpu().numpy()  # 节点坐标
        selected_node_list = self.selected_node_list[batch_idx, pomo_idx].cpu().numpy()  # 选择的节点列表
        node_service_time = self.node_service_time[batch_idx, pomo_idx].cpu().numpy()  # 服务时间
        timestamps = self.timestamps[batch_idx, pomo_idx].cpu().numpy()  # 时间戳
        tw_start = self.node_tw_start[batch_idx].cpu().numpy()  # 时间窗口开始
        tw_end = self.node_tw_end[batch_idx].cpu().numpy()  # 时间窗口结束
        if hasattr(self, 'node_profit'):
            node_profit = self.node_profit[batch_idx].cpu().numpy()
        else:
            node_profit = np.ones(len(node_xy))  # 如果没有收益信息，默认为1
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f'Orienteering Problem Solution (Batch {batch_idx}, POMO {pomo_idx})', fontsize=15)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        for i, (x, y) in enumerate(node_xy):
            if i == 0:  # 仓库节点用特殊标记
                ax.scatter(x, y, s=200, c='red', marker='*', edgecolors='black', zorder=5, label='Depot')
                ax.text(x, y+0.02, f'Depot', ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                ax.scatter(x, y, s=100, c='skyblue', edgecolors='black', zorder=3)
                profit_text = f"{i}\nP:{node_profit[i]:.1f}"
                tw_text = f"TW:[{tw_start[i]:.1f}, {tw_end[i]:.1f}]"
                ax.text(x, y+0.02, profit_text, ha='center', va='bottom', fontsize=8)
                ax.text(x, y-0.05, tw_text, ha='center', va='top', fontsize=7)
        routes = []
        current_route = []
        for node in selected_node_list:
            current_route.append(node)
            if node == 0 and len(current_route) > 1:  # 当遇到仓库节点且路径长度>1时
                routes.append(current_route)
                current_route = [0]  # 开始新路径，从仓库开始
        
        if len(current_route) > 1:
            routes.append(current_route)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, route in enumerate(routes):
            color = colors[i % len(colors)]
            route_x = [node_xy[node][0] for node in route]
            route_y = [node_xy[node][1] for node in route]
            
            ax.plot(route_x, route_y, c=color, linewidth=2, alpha=0.7, 
                    label=f'Route {i+1}', zorder=2)
            for j in range(len(route)-1):
                node = route[j]
                x, y = node_xy[node]
                global_idx = np.where(selected_node_list == node)[0][j if node == 0 else 0]
                service_time = node_service_time[global_idx] if global_idx < len(node_service_time) else 0
                timestamp = timestamps[global_idx] if global_idx < len(timestamps) else 0
                if j > 0 or i == 0:
                    marker_text = f"{j}" if j > 0 else "S/E"
                    ax.scatter(x, y, s=150, facecolors='white', edgecolors=color, 
                            linewidth=2, zorder=4)
                    ax.text(x, y, marker_text, ha='center', va='center', fontsize=9, 
                        fontweight='bold', color=color)
                    if node != 0:
                        time_text = f"Arr:{timestamp:.1f}\nSrv:{service_time:.1f}"
                        ax.text(x+0.05, y, time_text, ha='left', va='center', 
                            fontsize=7, color=color, alpha=0.9,
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Solution visualization saved to {save_path}")
        return fig, ax

def gen_tw(size, graph_size, time_factor, dura_region, rnds):
    """
    Copyright (c) 2020 Jonas K. Falkner
    Copy from https://github.com/jokofa/JAMPR/blob/master/data_generator.py
    """

    service_window = int(time_factor * 2)

    horizon = np.zeros((size, graph_size, 2))
    horizon[:] = [0, service_window]
    tw_start = rnds.randint(horizon[..., 0], horizon[..., 1] / 2)
    tw_start[:, 0] = 0
    epsilon = rnds.uniform(dura_region[0], dura_region[1], (tw_start.shape))
    duration = np.around(time_factor * epsilon)
    duration[:, 0] = service_window
    tw_end = np.minimum(tw_start + duration, horizon[..., 1]).astype(int)
    tw = np.concatenate([tw_start[..., None], tw_end[..., None]], axis=2).reshape(size, graph_size, 2)
    return tw

def generate_optwvp_data(size, graph_size, rnds=None, time_factor=100.0, loc_factor=100, tw_duration="5075"):
    """
    Copyright (c) 2020 Jonas K. Falkner
    Copy from https://github.com/jokofa/JAMPR/blob/master/data_generator.py
    """
    rnds = np.random if rnds is None else rnds
    service_window = int(time_factor * 2)
    nloc = rnds.uniform(size=(size, graph_size, 2)) * loc_factor  # node locations
    dura_region = {
         "5075": [.5, .75],
         "1020": [.1, .2],
    }
    if isinstance(tw_duration, str):
        dura_region = dura_region[tw_duration]
    else:
        dura_region = tw_duration
    tw = gen_tw(size, graph_size, time_factor, dura_region, rnds)
    return OPTWVP_SET(node_loc=nloc,
                     node_tw=tw,
                     durations=tw[..., 1] - tw[..., 0],
                     service_window=[service_window] * size,
                     time_factor=[time_factor] * size,
                     loc_factor=[loc_factor] * size, )

