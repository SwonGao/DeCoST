import pytz
import argparse
import pprint as pp
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import wandb
from Trainer import Trainer
from utils import *

def args2dict(args):
    env_params = {"problem_size": args.problem_size, "num_vehicles": args.num_vehicles, "pomo_size": args.pomo_size, "hardness": args.hardness,
                  "pomo_start": args.pomo_start, "val_dataset": args.val_dataset, "val_episodes": args.val_episodes,
                  "k_sparse": args.k_sparse, "max_tw_size": args.timewindows}

    model_params = {
                    # original parameters in MvMOE for POMO
                    "embedding_dim": args.embedding_dim, "sqrt_embedding_dim": args.sqrt_embedding_dim,
                    "encoder_layer_num": args.encoder_layer_num, "decoder_layer_num": args.decoder_layer_num,
                    "qkv_dim": args.qkv_dim, "head_num": args.head_num, "logit_clipping": args.logit_clipping,
                    "ff_hidden_dim": args.ff_hidden_dim, "norm": args.norm, "norm_loc": args.norm_loc,
                    "eval_type": args.eval_type, "problem": args.problem,
                    "tw_normalize": args.tw_normalize, 
                    "decision_boundary": args.decision_boundary, "detach_from_encoder": args.detach_from_encoder,
                    "use_ninf_mask_in_sl_MHA": args.use_ninf_mask_in_sl_MHA,
                    }

    optimizer_params = {"optimizer": {"lr": args.lr, "weight_decay": args.weight_decay},
                        "scheduler": {"milestones": args.milestones, "gamma": args.gamma}}

    trainer_params = {"epochs": args.epochs, "train_episodes": args.train_episodes, "accumulation_steps": args.accumulation_steps,
                      "train_batch_size": args.train_batch_size, "validation_interval": args.validation_interval,
                      "validation_batch_size": args.validation_batch_size, "model_save_interval": args.model_save_interval,
                      "timeout_reward": args.timeout_reward, "timeout_node_reward": args.timeout_node_reward,
                      "fsb_dist_only": args.fsb_dist_only, "fsb_reward_only": args.fsb_reward_only,
                      "penalty_increase": args.penalty_increase, "penalty_factor": args.penalty_factor,
                      "checkpoint": args.checkpoint, "load_optimizer": args.load_optimizer,
                      "decision_boundary": args.decision_boundary, "sl_loss":args.sl_loss,
                      "label_balance_sampling": args.label_balance_sampling, "fast_label_balance": args.fast_label_balance,
                      "fast_weight": args.fast_weight,
                      }

    return env_params, model_params, optimizer_params, trainer_params

# inference time is the average solution time
# optimization runtime is the second stage calculation time of a single batch
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A learning-based two-stage DEcoupled discrete-Continuous optimization with Service-time-guided Trajectory (DeCoST)")
    # env_params
    parser.add_argument('--problem', type=str, default="OPTWVP", choices=["OPTWVP"])
    parser.add_argument('--hardness', type=str, default="hard", choices=["hard"])
    parser.add_argument('--pomo_size', type=int, default=50, help="the number of start node, should <= problem size")
    parser.add_argument('--num_vehicles', type=int, default=3)
    parser.add_argument('--pomo_start', type=bool, default=False)
    parser.add_argument('--val_dataset', type=str, nargs='+', default=None, help="use the default one if set to None")
    # model_params
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--sqrt_embedding_dim', type=float, default=128**(1/2))
    parser.add_argument('--encoder_layer_num', type=int, default=6, help="the number of MHA in encoder")
    parser.add_argument('--decoder_layer_num', type=int, default=1, help="the number of MHA in decoder")
    parser.add_argument('--qkv_dim', type=int, default=16)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--logit_clipping', type=float, default=10)
    parser.add_argument('--ff_hidden_dim', type=int, default=512)
    parser.add_argument('--eval_type', type=str, default="argmax", choices=["argmax", "softmax"])
    parser.add_argument('--norm', type=str, default="instance", choices=["batch", "batch_no_track", "instance", "layer", "rezero", "none"])
    parser.add_argument('--norm_loc', type=str, default="norm_last", choices=["norm_first", "norm_last"], help="whether conduct normalization before MHA/FFN/MOE")
    
    # model
    parser.add_argument('--tw_normalize', type=bool, default=True)
    parser.add_argument('--k_sparse', type=int, default=500)
    parser.add_argument('--detach_from_encoder', type=bool, default=False)
    parser.add_argument('--use_ninf_mask_in_sl_MHA', type=bool, default= False)

    # optimizer_params
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--milestones', type=int, nargs='+', default=[9001, ], help='when to decay lr')
    parser.add_argument('--gamma', type=float, default=0.1, help='new_lr = lr * gamma')

    # trainer_params
    parser.add_argument('--epochs', type=int, default=800, help="total training epochs")
    parser.add_argument('--accumulation_steps', type=int, default=1)

    # batch and epoch
    parser.add_argument('--train_episodes', type=int, default=5000, help="the num. of training instances per epoch")
    parser.add_argument('--val_episodes', type=int, default=5000)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--validation_batch_size', type=int, default=128)
    parser.add_argument('--problem_size', type=int, default=50)
    parser.add_argument('--timewindows', type=int, default=100)

    parser.add_argument('--validation_interval', type=int, default=10)
    parser.add_argument('--model_save_interval', type=int, default=50)
    # reward params
    parser.add_argument('--timeout_reward', type=bool, default=True)
    parser.add_argument("--timeout_node_reward",type=bool,default=True)
    parser.add_argument('--fsb_dist_only', type=bool, default=True)
    parser.add_argument('--fsb_reward_only', type=bool, default=True) # activate only if no penalty
    parser.add_argument('--penalty_increase', type=bool, default=False)
    parser.add_argument('--penalty_factor', type=float, default=1.)
    # resume params
    parser.add_argument('--resume_path', type=str, default=None, help='path to the old run')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--load_optimizer', type=bool, default=True)
    # sl loss params
    parser.add_argument('--decision_boundary', type=float, default=0.5)
    parser.add_argument('--sl_loss', type=str, default="BCEWithLogitsLoss", choices=["BCEWithLogitsLoss", "BCELoss", "FL", "CE"], help="FL: focal loss; CE: cross entropy loss")
    parser.add_argument('--label_balance_sampling', type=bool, default=True)
    parser.add_argument('--fast_label_balance', type=bool, default=True)
    parser.add_argument('--fast_weight', type=bool, default=True)

    # settings (e.g., GPU)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--log_dir', type=str, default="./results")
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument("--multiple_gpu", type=bool, default=False)
    parser.add_argument('--occ_gpu', type=float, default=0., help="occupy (X)% GPU memory in advance, please use sparingly.")
    parser.add_argument('--tb_logger', type=bool, default=True)
    parser.add_argument('--wandb_logger', type=bool, default=True)

    args = parser.parse_args()
    seed_everything(args.seed)

    if args.val_dataset is None:
        args.val_dataset = [f"{args.problem.lower()}{args.problem_size}_{args.hardness}_{args.timewindows}.pkl"]

    # set log
    run_name = f"_{args.problem}{args.problem_size}_{args.hardness}"
    if args.timeout_reward:
        run_name += "_LM"
    process_start_time = datetime.now(pytz.timezone("Asia/Singapore"))
    if args.resume_path is not None:
        args.log_path = args.resume_path
    else:
        name = process_start_time.strftime("%Y%m%d_%H%M%S") + run_name
        args.log_path = os.path.join(args.log_dir, name)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    print(">> Log Path: {}".format(args.log_path))
    # set gpu
    if args.multiple_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not args.no_cuda and torch.cuda.is_available():
        if args.multiple_gpu and torch.cuda.device_count() > 1:
            args.device = torch.device('cuda')
        else:
            occumpy_mem(args) if args.occ_gpu != 0. else print(">> No occupation needed")
            # args.device = torch.device('cuda')
            args.gpu_id = int(args.gpu_id)
            args.device = torch.device('cuda', args.gpu_id)
            torch.cuda.set_device(args.gpu_id)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        args.device = torch.device('cpu')
    print(">> USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(not args.no_cuda, args.gpu_id))

    pp.pprint(vars(args))
    env_params, model_params, optimizer_params, trainer_params = args2dict(args)

    if args.wandb_logger:
        wandb.init(project="DeCoST", name=name,
                   config={**env_params, **model_params, **optimizer_params, **trainer_params})
    create_logger(filename="run_log", log_path=args.log_path)
    torch.set_printoptions(threshold=1000000)

    # start training
    print(">> Start {} Training ...".format(args.problem))

    trainer = Trainer(args=args, env_params=env_params, model_params=model_params, optimizer_params=optimizer_params, trainer_params=trainer_params)
    copy_all_src(args.log_path)
    trainer.run()

    print(">> Finish {} Training ...".format(args.problem))

