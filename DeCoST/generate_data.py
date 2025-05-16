import argparse
import pprint as pp
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets")
    parser.add_argument('--problem', type=str, default="OPTWVP", choices=["OPTWVP"])
    parser.add_argument('--problem_size', type=int, default=50)
    parser.add_argument('--pomo_size', type=int, default=50, help="the number of start node, should <= problem size")
    parser.add_argument('--hardness', type=str, default="hard", choices=["hard"], help="Different levels of constraint hardness")
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--dir', type=str, default="./data")
    parser.add_argument('--no_cuda', action='store_true', default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--timewindows', type=int, default=500)

    args = parser.parse_args()
    pp.pprint(vars(args))
    env_params = {"problem_size": args.problem_size, "pomo_size": args.pomo_size, "hardness": args.hardness}
    seed_everything(args.seed)

    # set log & gpu
    if not args.no_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda', args.gpu_id)
        torch.cuda.set_device(args.gpu_id)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        args.device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
    print(">> SEED: {}, USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(args.seed, not args.no_cuda, args.gpu_id))

    envs = get_env(args.problem)
    env_params = {'problem':args.problem, 'hardness':args.hardness, 
                  'problem_size': args.problem_size, 'pomo_size': args.pomo_size, 
                  'max_tw_size': args.timewindows}
    for env in envs:
        env = env(**env_params)
        if args.problem in ["OPTWVP"]:
            dataset_path = os.path.join(args.dir, env.problem, "{}{}_{}_{}.pkl".format(env.problem.lower(), args.problem_size, args.hardness, args.timewindows))
        else:
            dataset_path = os.path.join(args.dir, env.problem, "{}{}_uniform.pkl".format(env.problem.lower(), args.problem_size))

        env.generate_dataset(args.num_samples, args.problem_size, dataset_path)
        # sanity check
        data = env.load_dataset(dataset_path, num_samples=args.num_samples, disable_print=False)
        for i in range(len(data)):
            print(data[i][0])

