import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time  

class TOPTWVP_Gurobi_Solver:
    def __init__(self, max_distance, node_xy, service_time, tw_start, tw_end, s_tw_start, s_tw_end, profit, num_vehicles = None):
        """
        Initialize the TOPTWVP solver with problem parameters
        Args:
            max_distance: tensor[batchsize, 1]
            node_xy: tensor[batchsize, problemsize, 2]
            service_time: tensor[batchsize, problemsize]
            tw_start: tensor[batchsize, problemsize]
            tw_end: tensor[batchsize, problemsize]
            s_tw_start: tensor[batchsize, problemsize]
            s_tw_end: tensor[batchsize, problemsize]
        """
        # Convert tensors to numpy arrays
        self.max_distance = max_distance.cpu().numpy()
        node_xy_np = node_xy.cpu().numpy()
        
        # Get dimensions
        self.batch_size = len(node_xy_np)
        self.original_n_nodes = len(node_xy_np[0])
        
        # Add depot node (same as first node) to all arrays
        depot_node = node_xy_np[:, 0:1, :]  # Shape: [batch_size, 1, 2]
        self.node_xy = np.concatenate([node_xy_np, depot_node], axis=1)
        
        # Process all other arrays similarly
        self.service_time = np.concatenate([service_time.cpu().numpy(), service_time.cpu().numpy()[:, 0:1]], axis=1)
        self.tw_start = np.concatenate([tw_start.cpu().numpy(), tw_start.cpu().numpy()[:, 0:1]], axis=1)
        self.tw_end = np.concatenate([tw_end.cpu().numpy(), tw_end.cpu().numpy()[:, 0:1]], axis=1)
        self.s_tw_start = np.concatenate([s_tw_start.cpu().numpy(), s_tw_start.cpu().numpy()[:, 0:1]], axis=1)
        self.s_tw_end = np.concatenate([s_tw_end.cpu().numpy(), s_tw_end.cpu().numpy()[:, 0:1]], axis=1)
        self.profit = np.concatenate([profit.cpu().numpy(), profit.cpu().numpy()[:, 0:1]], axis=1)
        if num_vehicles is None:
            self.num_vehicles = np.ones(self.batch_size, dtype=int)
        else:
            self.num_vehicles = num_vehicles
        self.n_nodes = self.original_n_nodes + 1
        
        self.travel_times = [self._calculate_travel_times(b) for b in range(self.batch_size)]
        print("Gurobi initialized with depot as start and end node")
        
    def _calculate_travel_times(self, batch_idx):
        """
        Calculate travel times between all pairs of nodes for a specific batch
        Args:
            batch_idx: index of the batch to process
        """
        if batch_idx % 1000 == 0:
            print(f"Getting travel_distance batch: {batch_idx} / {self.batch_size}")
        travel_times = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    travel_times[i,j] = np.sqrt(
                        np.sum((self.node_xy[batch_idx,i] - self.node_xy[batch_idx,j])**2)
                    )
        return travel_times

    def solve(self, num_routes=1):
        """
        Solve TOPTWVP problems for all instances in the batch
        Args:
            num_routes: Number of routes to compute (M in the formulation)
        Returns:
            list of solutions for each batch instance
        """
        batch_solutions = []
        batch_times = []

        for b in range(self.batch_size):
            # Solve individual instance
            if b % 1 == 0:
                print(f"Solving instance {b}...")
            start_time = time.time()
            solution = self._solve_single_instance(b, int(self.num_vehicles[b]))
            end_time = time.time()
            elapsed = end_time - start_time
            batch_times.append(elapsed)
            batch_solutions.append(solution)
        avg_time = sum(batch_times) / len(batch_times)
        print(f"\nAverage time in this batch: {avg_time:.3f} seconds")
        return batch_solutions
    
    def _solve_single_instance(self, batch_idx, num_routes):
        """
        Solve a single TOPTWVP instance
        Args:
            batch_idx: index of the batch instance to solve
            num_routes: number of routes to compute
        """
        # Create model
        model = gp.Model(f"TOPTWVP_batch_{batch_idx}")
        
        # Create variables
        x = model.addVars(self.n_nodes, self.n_nodes, num_routes, vtype=GRB.BINARY, name="x")
        y = model.addVars(self.n_nodes, num_routes, vtype=GRB.BINARY, name="y")
        s = model.addVars(self.n_nodes, num_routes, lb=0, ub=float(self.max_distance[batch_idx][0]), name="s")
        d = model.addVars(self.n_nodes, num_routes, name="d")
        
        # Set objective: Maximize total profit
        model.setObjective(
            gp.quicksum(self.profit[batch_idx,i] * d[i,m] for m in range(num_routes) for i in range(1, self.n_nodes-1)), GRB.MAXIMIZE
        )
        
        # Constraints
        # (1) Each route starts at node 1 and ends at node N
        for m in range(num_routes):
            model.addConstr(
                gp.quicksum(x[0,j,m] for j in range(1, self.n_nodes)) == 1
            )
            model.addConstr(
                gp.quicksum(x[i,self.n_nodes-1,m] for i in range(self.n_nodes-1)) == 1
            )
        
        # (2) Each node can be visited at most once across all routes
        for k in range(1, self.n_nodes-1):
            model.addConstr(
                gp.quicksum(y[k,m] for m in range(num_routes)) <= 1
            )
        
        # (3) Route connectivity
        for k in range(1, self.n_nodes-1):
            for m in range(num_routes):
                model.addConstr(
                    gp.quicksum(x[i,k,m] for i in range(self.n_nodes-1)) ==
                    gp.quicksum(x[k,j,m] for j in range(1, self.n_nodes))
                )
                model.addConstr(
                    gp.quicksum(x[i,k,m] for i in range(self.n_nodes-1)) == y[k,m]
                )
        
        # (4) Time window constraints
        # IMPORTANT!!! will cause infeasible
        big_M = (self.max_distance[batch_idx][0])
        for i in range(self.n_nodes):
            for m in range(num_routes):
                model.addConstr(x[i,i,m] == 0)
                model.addConstr(s[i,m] >= self.tw_start[batch_idx,i] * y[i,m])
                model.addConstr(s[i,m] <= self.tw_end[batch_idx,i] * y[i,m] + big_M * (1 - y[i,m]))
        
        # (5) Timeline constraints
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    for m in range(num_routes):
                        model.addConstr(
                            s[i,m] + d[i,m] + self.travel_times[batch_idx][i,j] - s[j,m] <=
                            big_M * (1 - x[i,j,m])
                        )
        
        # (6) Service time interval constraints
        for i in range(self.n_nodes):
            for m in range(num_routes):
                model.addConstr(d[i,m] >= 0)
                model.addConstr(d[i,m] <= (self.tw_end[batch_idx,i] - self.tw_start[batch_idx,i]) * y[i,m])
        
        # Set solver parameters
        # model.setParam('OutputFlag', 0)   # Suppress output for batch processing
        # model.setParam('MIPGap', 0.01)   # 1% optimality gap
        # model.setParam('TimeLimit', 3600)  # 5 minute time limit
        # Optimize
        try:
            model.optimize()
            # print(model.display())

            # Extract solution
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                solution = {
                    'routes': [[] for _ in range(num_routes)],
                    'schedule': [[] for _ in range(num_routes)],
                    'service_times': [[] for _ in range(num_routes)],
                    'tw_start': [[] for _ in range(num_routes)],
                    'tw_end': [[] for _ in range(num_routes)],
                    's_tw_start': [[] for _ in range(num_routes)],
                    's_tw_end': [[] for _ in range(num_routes)],
                    'profit': [[] for _ in range(num_routes)],
                    'max_time': self.max_distance[batch_idx][0],
                    'rho': [[] for _ in range(num_routes)],  # Rho values for each route
                    'objective': model.objVal,
                    'status': model.status
                }
                
                # Extract routes
                for m in range(num_routes):
                    current_node = 0
                    # for i in range(self.n_nodes):
                    #     for j in range(self.n_nodes):
                    #         if x[i,j,m].x > 0.5:
                    #             print(i,j)

                    while current_node != self.n_nodes - 1:
                        solution['routes'][m].append(current_node)
                        solution['schedule'][m].append(s[current_node,m].x)
                        solution['service_times'][m].append(d[current_node,m].x)
                        solution['tw_start'][m].append(self.tw_start[batch_idx, current_node])
                        solution['tw_end'][m].append(self.tw_end[batch_idx, current_node])
                        solution['s_tw_start'][m].append(self.s_tw_start[batch_idx, current_node])
                        solution['s_tw_end'][m].append(self.s_tw_end[batch_idx, current_node])
                        solution['profit'][m].append(self.profit[batch_idx, current_node])
                        # Find next node
                        for j in range(self.n_nodes):
                            if x[current_node,j,m].x > 0.5:
                                current_node = j
                                break
                    
                    solution['routes'][m].append(self.n_nodes - 1)
                    solution['schedule'][m].append(s[self.n_nodes-1,m].x)
                    solution['service_times'][m].append(d[self.n_nodes-1,m].x)
                    solution['tw_start'][m].append(self.tw_start[batch_idx, self.n_nodes-1])
                    solution['tw_end'][m].append(self.tw_end[batch_idx, self.n_nodes-1])
                    solution['s_tw_start'][m].append(self.s_tw_start[batch_idx, self.n_nodes-1])
                    solution['s_tw_end'][m].append(self.s_tw_end[batch_idx, self.n_nodes-1])
                    solution['profit'][m].append(self.profit[batch_idx, self.n_nodes-1])
                    # solution['rho'][m].append(self.calculate_rho(solution))
                solution['rho'] = list([sum(values) for values in solution['service_times']] / solution['max_time'])
                # print("sb")
                return solution
            else:
                print(model.display())
                return {
                    'status': model.status,
                    'objective': None,
                    'routes': None,
                    'schedule': None,
                    'service_times': None,
                    'tw_start': None,
                    'tw_end': None,
                    's_tw_start': None,
                    's_tw_end': None,
                    'profit': None,
                    'max_time': None
                }
                
        except gp.GurobiError as e:
            print(f"Error in batch {batch_idx}: {e}")
            return None

    def save_solutions(self, batch_solutions, filename='../data/OPTWVP/gurobi_optwvp50_hard.pkl'):
        """
        Save solutions in the required format
        """
        import pickle
        import os
        
        formatted_solutions = []
        
        for batch_idx, solution in enumerate(batch_solutions):
            if solution['status'] == GRB.OPTIMAL:
                # Calculate optimal value
                opt_val = solution['objective']
                all_routes = solution['routes']
                rho = solution['rho']
                formatted_solutions.append([opt_val, all_routes, rho])
            else:
                # If no solution found, append None or appropriate placeholder
                formatted_solutions.append([0, [], 0])
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save to pickle file
        with open(filename, 'wb') as f:
            pickle.dump(formatted_solutions, f)
        
        return formatted_solutions

    def get_solution_summary(self, batch_solutions):
        """
        Generate a summary of the solutions for all batches
        Args:
            batch_solutions: list of solutions for each batch instance
        """
        summaries = []
        
        for b, solution in enumerate(batch_solutions):
            if solution is None:
                summaries.append(f"Batch {b}: Failed to solve")
                continue
                
            summary = [f"Batch {b}:"]
            if solution['objective'] is not None:
                summary.append(f"Objective value: {solution['objective']:.2f}")
                summary.append(f"Status: {solution['status']}")
                
                for m, route in enumerate(solution['routes']):
                    route_str = f"Route {m+1}: "
                    for i, node in enumerate(route):
                        route_str += f"Node {node} (arr: {solution['schedule'][m][i]:.2f}, "
                        route_str += f"srv: {solution['service_times'][m][i]:.2f}) → "
                    summary.append(route_str[:-2])
            else:
                summary.append(f"Status: {solution['status']}")
                
            summaries.append("\n".join(summary))
            
        return "\n\n".join(summaries)