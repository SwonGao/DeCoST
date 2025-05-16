import pickle
import os
import torch
from gurobipy import GRB 
# from POMO_onestage.envs import OPTWVPEnv
from envs import OPTWVPEnv
from TwoPhaseOPTWVPSolver import TwoPhaseOPTWVPSolver_0service, TwoPhaseOPTWVPSolver_50service, TwoPhaseOPTWVPSolver_100service
import pickle
import os
from gurobipy import GRB 


def convert_pkl_to_txt(pkl_path, output_dir, dataset):
    # Load the pkl file
    node_xy, service_time, tw_start, tw_end, profit, s_tw_start, s_tw_end, max_travel_distance = dataset
    # node_xy, service_time, tw_start, tw_end, profit, s_tw_start, s_tw_end, max_travel_distance = \
    #     torch.Tensor(node_xy), torch.Tensor(service_time), torch.Tensor(tw_start), torch.Tensor(tw_end), \
    #     torch.Tensor(profit), torch.Tensor(s_tw_start), torch.Tensor(s_tw_end), torch.Tensor(max_travel_distance)

    # Convert tensors to lists
    node_xy = node_xy.tolist()
    tw_start = tw_start.tolist()
    tw_end = tw_end.tolist()
    profit = profit.tolist()
    max_travel_distance = max_travel_distance.tolist()
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate 20 output files
    for idx in range(5):
        txt_path = os.path.join(output_dir, f"optwvp50_hard_{tw_size}_{idx+1}.txt")
        with open(txt_path, 'w') as f:
            f.write("C101\n\n")
            f.write("VEHICLE\n")
            f.write("NUMBER     CAPACITY\n")
            f.write(f"1        {int(max_travel_distance[idx] * 100.) }\n\n")
            
            f.write("CUSTOMER\n")
            f.write("CUST NO.   XCOORD.   YCOORD.   DEMAND   READY TIME   DUE DATE   SERVICE TIME\n\n")
            
            for i in range(len(node_xy[idx])):
                f.write(
                    f" {i:<10} {float(node_xy[idx][i][0]/1.):<10} {float(node_xy[idx][i][1]/1.):<10} {float(profit[idx][i]):<10} "
                    f"{float(tw_start[idx][i]/1.):<12} {float(tw_end[idx][i]/1.):<10} 1\n"
                )
        print(f"Converted {pkl_path} to {txt_path}")

def get_env(problem):
    from envs import TOPTWVPEnv, TOPTWEnv, OPTWEnv, TSPDLEnv, TSPTWEnv, OPTWEnvsb
    all_problems = {
        'TOPTWVP': TOPTWVPEnv.TOPTWVPEnv,
        'OPTWVP': OPTWVPEnv.OPTWVPEnv,
        'TOPTW': TOPTWEnv.TOPTWEnv,
        'OPTW': OPTWEnvsb.OPTWEnv,
        'TSPTW': TSPTWEnv.TSPTWEnv,
        'TSPDL': TSPDLEnv.TSPDLEnv,
    }
    if problem == "ALL":
        return list(all_problems.values())
    else:
        return [all_problems[problem]]

def get_opt_sol_path(dir, problem, size, hardness, tw_size):
    if problem in ["TOPTW", "TSPTW", "TSPDL"]:
        return os.path.join(dir, f"lkh_{problem.lower()}{size}_{hardness}_{tw_size}.pkl")
    elif problem in ["OPTWVP"]:
        if hardness=="hard":
            return os.path.join(dir, f"gurobi_optwvp50_{hardness}_{tw_size}.pkl")
        else:
            return os.path.join(dir, f"gurobi_optwvp50_{hardness}.pkl")
    else:
        all_opt_sol = {
            'CVRP': {50: 'hgs_cvrp50_uniform.pkl', 100: 'hgs_cvrp100_uniform.pkl'},
            'OVRP': {50: 'or_tools_200s_ovrp50_uniform.pkl', 100: 'lkh_ovrp100_uniform.pkl'},
            'VRPB': {50: 'or_tools_200s_vrpb50_uniform.pkl', 100: 'or_tools_400s_vrpb100_uniform.pkl'},
            'VRPL': {50: 'or_tools_200s_vrpl50_uniform.pkl', 100: 'lkh_vrpl100_uniform.pkl'},
            'VRPTW': {50: 'hgs_vrptw50_uniform.pkl', 100: 'hgs_vrptw100_uniform.pkl'},
            'OVRPTW': {50: 'or_tools_200s_ovrptw50_uniform.pkl', 100: 'or_tools_400s_ovrptw100_uniform.pkl'},
            'OVRPB': {50: 'or_tools_200s_ovrpb50_uniform.pkl', 100: 'or_tools_400s_ovrpb100_uniform.pkl'},
            'OVRPL': {50: 'or_tools_200s_ovrpl50_uniform.pkl', 100: 'or_tools_400s_ovrpl100_uniform.pkl'},
            'VRPBL': {50: 'or_tools_200s_vrpbl50_uniform.pkl', 100: 'or_tools_400s_vrpbl100_uniform.pkl'},
            'VRPBTW': {50: 'or_tools_200s_vrpbtw50_uniform.pkl', 100: 'or_tools_400s_vrpbtw100_uniform.pkl'},
            'VRPLTW': {50: 'or_tools_200s_vrpltw50_uniform.pkl', 100: 'or_tools_400s_vrpltw100_uniform.pkl'},
            'OVRPBL': {50: 'or_tools_200s_ovrpbl50_uniform.pkl', 100: 'or_tools_400s_ovrpbl100_uniform.pkl'},
            'OVRPBTW': {50: 'or_tools_200s_ovrpbtw50_uniform.pkl', 100: 'or_tools_400s_ovrpbtw100_uniform.pkl'},
            'OVRPLTW': {50: 'or_tools_200s_ovrpltw50_uniform.pkl', 100: 'or_tools_400s_ovrpltw100_uniform.pkl'},
            'VRPBLTW': {50: 'or_tools_200s_vrpbltw50_uniform.pkl', 100: 'or_tools_400s_vrpbltw100_uniform.pkl'},
            'OVRPBLTW': {50: 'or_tools_200s_ovrpbltw50_uniform.pkl', 100: 'or_tools_400s_ovrpbltw100_uniform.pkl'},
        }
        return os.path.join(dir, all_opt_sol[problem][size])

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def load_dataset(filename, disable_print=False):
    with open(check_extension(filename), 'rb') as f:
        data = pickle.load(f)
    if not disable_print:
        print(">> Load {} data ({}) from {}".format(len(data), type(data), filename))
    return data

def gurobi_solving_problem(hardness, test_num_episode, data_path, problem, problem_size, tw_size, offset):
## calculation of optimal solution and append it into the pkl
## if you need to calculate the optimal solution for TOPTWVP and output it to the pkl file, 
## please uncomment the following code
    envs = get_env(problem)  # Env Class
    for env_class in envs:
        env_params = {'hardness':hardness, 'problem_size': problem_size, 'pomo_size': 50, 'loc_scaler': None,
                        'device': 'cpu', "dl_percent":None, "reverse":None, "original_lib_xy": None,
                        'k_sparse': 500}
        env = env_class(**env_params)
    # solve the optimal solution
    dataset_solutions = []
    aug_factor = 1
    test_data = env.load_dataset(data_path, offset=offset, num_samples=test_num_episode)
    env.load_problems(test_num_episode, problems=test_data, aug_factor=aug_factor, normalize=True)
    solution = env.get_gurobi_optimal_solution()
    dataset_solutions.append(solution)
    # format 
    if env.problem == "OPTWVP":
        filename=f"../data/OPTWVP/gurobi_optwvp{problem_size}_{env.hardness}_{tw_size}_{offset}.pkl"
    elif env.problem == "OPTW":
        filename=f"../data/OPTW/gurobi_optw{problem_size}_{env.hardness}_{tw_size}_{offset}.pkl"
    elif env.problem == "TOPTWVP":
        filename=f"../data/TOPTWVP/gurobi_toptwvp{problem_size}_{env.hardness}_{tw_size}_{offset}.pkl"

    formatted_solutions = []
    for batch_idx, solutions in enumerate(dataset_solutions):
            # Calculate optimal value
        for num, solution in enumerate(solutions):
            if solution['status'] == GRB.OPTIMAL or solution['status'] == GRB.TIME_LIMIT:
                opt_val = solution['objective']
                all_routes = solution['routes']
                rho = solution['rho']
                formatted_solutions.append([opt_val, all_routes, rho])
            else:
                # If no solution found, append None or appropriate placeholder
                formatted_solutions.append([-1, [], 0]) 
                print("no optimal solution if found! ")

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(formatted_solutions, f)        
    print(">> Optimal solution is saved to {}".format(filename))

def gurobi_solving_problem2(hardness, test_num_episode, data_path, problem, problem_size, tw_size, batch_size=100):
    ## calculation of optimal solution and append it into the pkl
    envs = get_env(problem)  # Env Class
    for env_class in envs:
        env_params = {'hardness':hardness, 'problem_size': problem_size, 'pomo_size': 50, 'loc_scaler': None,
                      'device': 'cpu', "dl_percent":None, "reverse":None, "original_lib_xy": None,
                      'k_sparse': 500}
        env = env_class(**env_params)
    
    # 计算需要的批次数
    num_batches = (test_num_episode + batch_size - 1) // batch_size  # 向上取整
    
    # 对每个批次进行处理
    for batch_idx in range(num_batches):
        print(f"Processing batch {batch_idx+1}/{num_batches}")
        
        # 计算当前批次的offset和实际样本数
        offset = batch_idx * batch_size
        current_batch_size = min(batch_size, test_num_episode - offset)
        
        # 加载当前批次的数据
        test_data = env.load_dataset(data_path, offset=offset, num_samples=current_batch_size)
        env.load_problems(current_batch_size, problems=test_data, aug_factor=1, normalize=True)
        
        # 求解当前批次
        solution = env.get_gurobi_optimal_solution()
        
        # 处理当前批次的结果
        batch_formatted_solutions = []
        for sol in solution:
            if sol['status'] == GRB.OPTIMAL or sol['status'] == GRB.TIME_LIMIT:
                opt_val = sol['objective']
                all_routes = sol['routes']
                rho = sol['rho']
                batch_formatted_solutions.append([opt_val, all_routes, rho])
            else:
                # 如果没有找到解，添加占位符
                batch_formatted_solutions.append([-1, [], 0])
                print("No optimal solution found for an instance in the current batch!")
        
        # 不再需要将结果添加到总列表中，因为每个批次单独保存
        
        # 每个批次都保存为单独的文件
        # 确定当前批次的文件名
        if env.problem == "OPTWVP":
            filename = f"../data/OPTWVP/gurobi_optwvp{env.problem_size}_{env.hardness}_{tw_size}_{batch_idx}.pkl"
        elif env.problem == "OPTW":
            filename = f"../data/OPTW/gurobi_optw50_{env.hardness}_{tw_size}_{batch_idx}.pkl"
        elif env.problem == "TOPTWVP":
            filename = f"../data/TOPTWVP/gurobi_toptwvp50_{env.hardness}_{tw_size}_{batch_idx}.pkl"
        
        # 只保存当前批次的结果
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(batch_formatted_solutions, f)
        print(f">> Batch {batch_idx} results saved to {filename}")
    
    print(f">> All {num_batches} batches have been processed and saved separately.")

problem = "OPTWVP"
hardness = "hard"
tw_size = 100
test_num_episode = 100
problem_size = 500
if hardness == "hard":
    dataset_path = f"../data/{problem}/{problem.lower()}{problem_size}_{hardness}_{tw_size}.pkl"
else:
    dataset_path = f"..\\data\\{problem}\\{problem.lower()}{problem_size}_{hardness}.pkl"

gurobi_solving_problem(hardness, test_num_episode, dataset_path, problem, problem_size, tw_size, 0)

##################################################################################3
##################################################################################3
##################################################################################3
# try:
#     sol_path = get_opt_sol_path(f"..\\data\\{problem}", problem, 50, hardness, tw_size)
#     # sol_path=f"../data/OPTWVP/gurobi_optwvp50_hard.pkl"
# except:
#     sol_path = None

# compute_gap = os.path.exists(sol_path)
# if compute_gap:
#     opt_sol = load_dataset(sol_path, disable_print=True)[: test_num_episode]
# opt_val = ([i[0] for i in opt_sol])
# opt_seq = ([i[1] for i in opt_sol])
# opt_rho = ([i[2] for i in opt_sol])

# dataset_solutions = []

# envs = get_env("OPTWVP")  # Env Class
# for env_class in envs:
    # env_params = {'hardness':hardness, 'problem_size': 50, 'pomo_size': 50, 'loc_scaler': None,
                    # 'device': 'cpu', "dl_percent":None, "reverse":None, "original_lib_xy": None,
                    # 'k_sparse': 50}
    # env = env_class(**env_params)
# 
# test_data = env.load_dataset(dataset_path, offset=test_num_episode, num_samples=test_num_episode)
# env.load_problems(test_num_episode, problems=test_data, aug_factor=1, normalize=True)

# # solutions = env.get_gurobi_two_stage_solution_0service(opt_val, opt_rho)
# for percent in range(10, 101, 10):
#     solutions = env.get_gurobi_two_stage_solution_50service(opt_val, opt_rho, percent)

# solutions = env.get_gurobi_two_stage_solution_50service(opt_val, opt_rho, 100)

##################################################################################3
##################################################################################3
##################################################################################3
## generate dataset for ILS
# dataset_path = f"optwvp50_hard_{tw_size}.pkl"
# output_directory = "output_files" 
# convert_pkl_to_txt(dataset_path, output_directory, test_data)