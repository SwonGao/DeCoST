from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from tensorboard_logger import Logger as TbLogger
from utils import *
from models.SINGLEModel import SINGLEModel
from sklearn.utils.class_weight import compute_class_weight
import os, wandb
from sklearn.metrics import confusion_matrix

class Trainer:
    def __init__(self, args, env_params, model_params, optimizer_params, trainer_params):
        # save arguments
        self.args = args
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.problem = self.args.problem
        self.penalty_factor = args.penalty_factor

        self.device = args.device
        self.log_path = args.log_path
        self.result_log = {"val_score": [], "val_gap": [], "val_infsb_rate": []}
        if args.tb_logger:
            self.tb_logger = TbLogger(self.log_path)
        else:
            self.tb_logger = None
        self.wandb_logger = args.wandb_logger

        # Main Components
        self.envs = get_env(self.args.problem)  # a list of envs classes (different problems), remember to initialize it!
        self.model = SINGLEModel(**self.model_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        num_param(self.model)
        # Restore
        self.start_epoch = 1
        if args.checkpoint is not None:
            checkpoint_fullname = args.checkpoint
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            except:
                self.model.load_state_dict(checkpoint, strict=True)

            self.start_epoch = 1 + checkpoint['epoch']
            self.scheduler.last_epoch = checkpoint['epoch'] - 1
            if self.trainer_params["load_optimizer"]:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(">> Optimizer (Epoch: {}) Loaded (lr = {})!".format(checkpoint['epoch'], self.optimizer.param_groups[0]['lr']))
            print(">> Checkpoint (Epoch: {}) Loaded!".format(checkpoint['epoch']))
            print(">> Load from {}".format(checkpoint_fullname))

        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            print('================================================================================')

            if self.trainer_params["penalty_increase"]:
                self.penalty_factor = 0.5 + epoch / self.trainer_params["epochs"] * 1.5

            train_score, train_loss, train_sl_st_loss, infeasible = self._train_one_epoch(epoch)

            self.scheduler.step()

            if isinstance(train_score, list):
                dist_reward, total_timeout_reward, timeout_nodes_reward = train_score
                train_score = dist_reward
            if self.trainer_params["fsb_dist_only"]:
                try:
                    sol_infeasible_rate, ins_infeasible_rate, feasible_dist_mean, feasible_dist_max_pomo_mean = infeasible
                except:
                    pass
            else:
                sol_infeasible_rate, ins_infeasible_rate = infeasible
            if self.tb_logger:
                self.tb_logger.log_value('train/train_score', train_score, epoch)
                self.tb_logger.log_value('train/train_loss', train_loss, epoch)
                self.tb_logger.log_value('train/train_sl_st_loss', train_sl_st_loss, epoch)
                try:
                    self.tb_logger.log_value('feasibility/solution_infeasible_rate', sol_infeasible_rate, epoch)
                    self.tb_logger.log_value('feasibility/instance_infeasible_rate', ins_infeasible_rate, epoch)
                except:
                    pass
                if self.trainer_params["timeout_reward"]:
                    self.tb_logger.log_value("feasibility/total_timeout", total_timeout_reward, epoch)
                    self.tb_logger.log_value("feasibility/timeout_nodes", timeout_nodes_reward, epoch)
                if self.trainer_params["fsb_dist_only"]:
                    self.tb_logger.log_value("feasibility/feasible_dist_mean", feasible_dist_mean, epoch)
                    self.tb_logger.log_value("feasibility/feasible_dist_max_pomo_mean", feasible_dist_max_pomo_mean, epoch)
            if self.wandb_logger:
                wandb.log({'train/train_score': train_score})
                wandb.log({'train/train_loss': train_loss})
                try:
                    wandb.log({'feasibility/solution_infeasible_rate': sol_infeasible_rate})
                    wandb.log({'feasibility/instance_infeasible_rate': ins_infeasible_rate})
                except:
                    pass
                if self.trainer_params["timeout_reward"]:
                    wandb.log({"feasibility/total_timeout": total_timeout_reward})
                    wandb.log({"feasibility/timeout_nodes": timeout_nodes_reward})
                if self.trainer_params["fsb_dist_only"]:
                    wandb.log({"feasibility/feasible_dist_mean": feasible_dist_mean})
                    wandb.log({"feasibility/feasible_dist_max_pomo_mean": feasible_dist_max_pomo_mean})

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            print("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['model_save_interval']
            validation_interval = self.trainer_params['validation_interval']

            try:
                if train_score < best_score:
                    best_score = train_score
                    torch.save(self.model.state_dict(), os.path.join(self.log_path, "trained_model_best.pt"))
                    print(">> Best model saved!")
            except:
                best_score = train_score

            # Save model
            if all_done or (epoch % model_save_interval == 0):
                print("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'problem': self.args.problem,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log,
                }
                torch.save(checkpoint_dict, '{}/epoch-{}.pt'.format(self.log_path, epoch))

            # validation
            if epoch == 1 or (epoch % validation_interval == 0):
                val_problems = [self.args.problem]
                val_episodes, problem_size = self.env_params['val_episodes'], self.env_params['problem_size']
                if self.env_params['val_dataset'] is not None:
                    paths = self.env_params['val_dataset']
                    dir = ["../data/{}/".format(self.args.problem)] * len(paths)
                    val_envs = [get_env(prob)[0] for prob in val_problems] * len(paths)
                else:
                    dir = [os.path.join("../data", prob) for prob in val_problems]
                    paths = ["{}{}_uniform.pkl".format(prob.lower(), problem_size) for prob in val_problems]
                    val_envs = [get_env(prob)[0] for prob in val_problems]
                for i, path in enumerate(paths):
                    # if no optimal solution provided, set compute_gap to False
                    if not self.env_params["pomo_start"]:
                        # sampling pomo_size routes is useless due to the argmax operator when selecting next node based on probability
                        init_pomo_size = self.env_params["pomo_size"]
                        self.env_params["pomo_size"] = 1
                    score, gap, infsb_rate = self._val_and_stat(dir[i], path, val_envs[i](**self.env_params), batch_size=self.trainer_params["validation_batch_size"], val_episodes=val_episodes, epoch = epoch)
                    if not self.env_params["pomo_start"]:
                        self.env_params["pomo_size"] = init_pomo_size
                    self.result_log["val_score"].append(score)
                    self.result_log["val_gap"].append(gap)
                    if infsb_rate is not None:
                        self.result_log["val_infsb_rate"].append(infsb_rate)
                    if self.tb_logger:
                        self.tb_logger.log_value('val_score/{}'.format(path.split(".")[0]), score, epoch)
                        self.tb_logger.log_value('val_gap/{}'.format(path.split(".")[0]), gap, epoch)
                        try:
                            self.tb_logger.log_value('val_sol_infsb_rate/{}'.format(path.split(".")[0]), infsb_rate[0], epoch)
                            self.tb_logger.log_value('val_ins_infsb_rate/{}'.format(path.split(".")[0]), infsb_rate[1], epoch)
                        except:
                            pass
                    if self.wandb_logger:
                        wandb.log({'val_score/{}'.format(path.split(".")[0]): score})
                        wandb.log({'val_gap/{}'.format(path.split(".")[0]): gap})
                        try:
                            wandb.log({'val_sol_infsb_rate/{}'.format(path.split(".")[0]): infsb_rate[0]})
                            wandb.log({'val_ins_infsb_rate/{}'.format(path.split(".")[0]): infsb_rate[1]})
                        except:
                            pass

                    try:
                        if score < best_val_score:
                            best_val_score = score
                            torch.save(self.model.state_dict(), os.path.join(self.log_path, "trained_model_val_best.pt"))
                            print(">> Best model on validation dataset saved!")
                    except:
                        best_val_score = score

    def _train_one_epoch(self, epoch):
        episode = 0
        score_AM, loss_AM, sl_st_loss_AM, sol_infeasible_rate_AM, ins_infeasible_rate_AM = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        if self.trainer_params["fsb_dist_only"]:
            feasible_dist_mean_AM, feasible_dist_max_pomo_mean_AM = AverageMeter(), AverageMeter()
        if self.trainer_params["timeout_reward"]:
            timeout_AM, timeout_nodes_AM = AverageMeter(), AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        total_step = math.floor(train_num_episode /self.trainer_params['train_batch_size'])
        batch_id = 0
        while episode < train_num_episode:
            for accumulation_step in range(self.trainer_params['accumulation_steps']):
                remaining = train_num_episode - episode
                batch_size = min(self.trainer_params['train_batch_size'], remaining)

                env = random.sample(self.envs, 1)[0](**self.env_params)
                data = env.get_random_problems(batch_size, self.env_params["problem_size"], 100, self.args.timewindows)

                avg_score, avg_loss, sl_st_loss, infeasible, sl_output = self._train_one_batch(data, env, accumulation_step=accumulation_step)

                if isinstance(infeasible, dict):
                    sol_infeasible_rate = infeasible["sol_infeasible_rate"]
                    ins_infeasible_rate = infeasible["ins_infeasible_rate"]
                    try:
                        feasible_dist_mean, feasible_dist_mean_num = infeasible["feasible_dist_mean"]
                        feasible_dist_max_pomo_mean, feasible_dist_max_pomo_mean_num = infeasible["feasible_dist_max_pomo_mean"]
                        feasible_dist_mean_AM.update(feasible_dist_mean, feasible_dist_mean_num)
                        feasible_dist_max_pomo_mean_AM.update(feasible_dist_max_pomo_mean, feasible_dist_max_pomo_mean_num)
                    except:
                        pass
                else:
                    infeasible_rate = infeasible

                if isinstance(avg_score, list):
                    dist_reward, total_timeout_reward, timeout_nodes_reward = avg_score
                    avg_score = dist_reward
                    timeout_AM.update(total_timeout_reward, batch_size)
                    timeout_nodes_AM.update(timeout_nodes_reward, batch_size)
                score_AM.update(avg_score, batch_size)
                loss_AM.update(avg_loss, batch_size)
                sl_st_loss_AM.update(sl_st_loss, batch_size)
                try:
                    sol_infeasible_rate_AM.update(sol_infeasible_rate, batch_size)
                    ins_infeasible_rate_AM.update(ins_infeasible_rate, batch_size)
                except:
                    pass

                episode += batch_size
                batch_id += 1
                if episode >= train_num_episode:
                    break

        if self.trainer_params["timeout_reward"]:
            print(
                'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}, sl_Loss: {:.4f}, Infeasible_rate: [{:.4f}%, {:.4f}%], Timeout: {:.4f}, Timeout_nodes: {:.0f}, Feasible_dist: {:.4f}'.format(
                    epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg, sl_st_loss_AM.avg,
                    sol_infeasible_rate_AM.avg * 100, ins_infeasible_rate_AM.avg * 100, timeout_AM.avg,
                    timeout_nodes_AM.avg, feasible_dist_max_pomo_mean_AM.avg))
        else:
            try:
                print('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},  Infeasible_rate: [{:.4f}%, {:.4f}%], Feasible_dist: {:.4f}'.format(epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg, sol_infeasible_rate_AM.avg*100, ins_infeasible_rate_AM.avg*100, feasible_dist_max_pomo_mean_AM.avg))
            except:
                print('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'.format(epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg))

        if self.trainer_params["fsb_dist_only"]:
            try:
                infeasible_output = [sol_infeasible_rate_AM.avg, ins_infeasible_rate_AM.avg, feasible_dist_mean_AM.avg, feasible_dist_max_pomo_mean_AM.avg]
            except:
                infeasible_output = None
        else:
            infeasible_output = [sol_infeasible_rate_AM.avg, ins_infeasible_rate_AM.avg]

        if self.trainer_params["timeout_reward"]:
            score_output = [score_AM.avg, timeout_AM.avg, timeout_nodes_AM.avg]
        else:
            score_output = score_AM.avg

        return score_output, loss_AM.avg, sl_st_loss_AM.avg, infeasible_output

    def _train_one_batch(self, data, env, accumulation_step):
        self.model.train()
        self.model.set_eval_type(self.model_params["eval_type"])
        batch_size = data.size(0) if isinstance(data, torch.Tensor) else data[-1].size(0)
        env.load_problems(batch_size, problems=data, aug_factor=1)
        batch_size = env.batch_size
        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state)
        prob_list = torch.zeros(size=(env.POMO_IDX.shape[0], env.POMO_IDX.shape[1], 0), device=env.device)

        state, reward, done = env.pre_step()
        while not done:
            selected, prob, service_time_normed = self.model(state, pomo=self.env_params["pomo_start"],
                                            tw_end = env.node_tw_end if self.problem in ["OPTWVP", "TOPTW", "OPTW", "TSPTW"] else None,
                                            no_sigmoid = (self.trainer_params["sl_loss"] == "BCEWithLogitsLoss"))

            state, reward, done, infeasible = env.step(selected,
                                                       service_time_normed = service_time_normed,
                                                       out_reward = self.trainer_params["timeout_reward"],
                                                       )
            if isinstance(infeasible, list):
                infeasible, infsb_level_value = infeasible
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        ########################################
        ############ Calculate Loss ############
        ########################################
        # 1. 计算奖励
        infeasible_output = infeasible
        if isinstance(reward, list):
            dist_reward, total_timeout_reward, timeout_nodes_reward, service_time_sl_reward = reward
            dist = dist_reward.clone()
        else:
            dist_reward = reward
            dist = reward
        if self.trainer_params["fsb_dist_only"]:
            problem_size, batch_size, pomo_size = self.env_params["problem_size"], env.batch_size, env.pomo_size
            feasible_number = (batch_size*pomo_size) - infeasible.sum()
            feasible_dist_mean, feasible_dist_max_pomo_mean = 0., 0.
            batch_feasible = torch.tensor([0.])
            if feasible_number:
                feasible_dist = torch.where(infeasible, torch.zeros_like(dist_reward), dist_reward) # feasible dist left only
                feasible_dist_mean = -feasible_dist.sum() / feasible_number # negative sign to make positive value, and calculate mean
                feasible_dist_mean = (feasible_dist_mean, feasible_number)
                reward_masked = dist.masked_fill(infeasible, -1e10)  # get feasible results from pomo
                feasible_max_pomo_dist = reward_masked.max(dim=1)[0]# get best results from pomo, shape: (batch)
                batch_feasible = (infeasible==False).any(dim=-1) # shape: (batch)
                feasible_max_pomo_dist = torch.where(batch_feasible==False, torch.zeros_like(feasible_max_pomo_dist), feasible_max_pomo_dist) # feasible dist left only
                feasible_dist_max_pomo_mean = -feasible_max_pomo_dist.sum() / batch_feasible.sum() # negative sign to make positive value, and calculate mean
                feasible_dist_max_pomo_mean = (feasible_dist_max_pomo_mean, batch_feasible.sum())
            
            infeasible_output = {
                "sol_infeasible_rate": infeasible.sum() / (batch_size*pomo_size),
                "ins_infeasible_rate": 1. - batch_feasible.sum() / batch_size,
                "feasible_dist_mean": feasible_dist_mean,
                "feasible_dist_max_pomo_mean": feasible_dist_max_pomo_mean
            }
        if isinstance(reward, list):
            reward = dist + self.penalty_factor * (total_timeout_reward + timeout_nodes_reward)  # (batch, pomo)
        if not self.trainer_params["timeout_reward"] and self.trainer_params["fsb_reward_only"]: # activate when not using LM
            feasible_reward_number = (infeasible==False).sum(-1)
            feasible_reward_mean = (torch.where(infeasible, torch.zeros_like(dist_reward), dist_reward).sum(-1) / feasible_reward_number)[:,None]
            feasible_advantage = dist_reward - feasible_reward_mean
            feasible_advantage = torch.masked_select(feasible_advantage, infeasible==False)
            log_prob = torch.masked_select(prob_list.log().sum(dim=2), infeasible==False)
            advantage = feasible_advantage
        else:
            advantage = reward - reward.float().mean(dim=1, keepdims=True)  # (batch, pomo)
            log_prob = prob_list.log().sum(dim=2)
        loss = - advantage * log_prob  # Minus Sign: To Increase REWARD
        loss_mean = loss.mean()
        sl_st_loss = - service_time_sl_reward.mean()
        loss_mean = loss_mean + sl_st_loss # service_time_sl_reward

        if accumulation_step == 0:
            self.model.zero_grad()

        loss_mean = loss_mean/self.trainer_params["accumulation_steps"]
        loss_mean.backward()
        if accumulation_step == self.trainer_params["accumulation_steps"] - 1:
            self.optimizer.step()
        if not self.trainer_params["timeout_reward"]:
            max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
            score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value
            score_mean = score_mean.item()
        else:
            max_dist_reward = dist_reward.max(dim=1)[0]  # get best results from pomo
            dist_mean = -max_dist_reward.float().mean()  # negative sign to make positive value
            max_timeout_reward = total_timeout_reward.max(dim=1)[0]  # get best results from pomo
            timeout_mean = -max_timeout_reward.float().mean()  # negative sign to make positive value
            max_timeout_nodes_reward = timeout_nodes_reward.max(dim=1)[0]  # get best results from pomo
            timeout_nodes_mean = -max_timeout_nodes_reward.float().mean()  # negative sign to make positive value
            score_mean = [dist_mean, timeout_mean, timeout_nodes_mean]

        loss_out = (loss_mean - sl_st_loss).item()
        sl_st_loss = sl_st_loss.item()
        return score_mean, loss_out, sl_st_loss, infeasible_output, None

    def _val_one_batch(self, data, env, aug_factor=1, eval_type="argmax"):
        self.model.eval()
        self.model.set_eval_type(eval_type)

        batch_size = data.size(0) if isinstance(data, torch.Tensor) else data[-1].size(0)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 确保GPU操作完成
        start_total_time = time.time()       
        with torch.no_grad():
            env.load_problems(batch_size, problems=data, aug_factor=aug_factor, normalize=True)
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            # print("load_env done!")
            while not done:      
                selected, prob, service_time_normed = self.model(state, pomo=self.env_params["pomo_start"],
                                               tw_end = env.node_tw_end if self.problem in ["OPTWVP"] else None
                                               )

                state, reward, done, infeasible = env.step(selected,
                                                           service_time_normed = service_time_normed
                                                           )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_total_time = time.time()
        total_inference_time = end_total_time - start_total_time

        if not hasattr(self, 'total_inference_times'):
            self.total_inference_times = []
        self.total_inference_times.append({
            'total_time': total_inference_time,
            'batch_size': batch_size
        })

        if isinstance(reward, list):
            dist_reward, total_timeout_reward, timeout_nodes_reward, service_time_sl_reward = reward
            dist = dist_reward.clone()

            aug_total_timeout_reward = total_timeout_reward.reshape(aug_factor, batch_size, env.pomo_size)
            # shape: (augmentation, batch, pomo)
            max_pomo_total_timeout_reward, _ = aug_total_timeout_reward.max(dim=2)  # get best results from pomo
            no_aug_total_timeout_score = -max_pomo_total_timeout_reward[0, :].float()  # negative sign to make positive value
            max_aug_pomo_total_timeout_reward, _ = max_pomo_total_timeout_reward.max(dim=0)  # get best results from augmentation
            aug_total_timeout_score = -max_aug_pomo_total_timeout_reward.float()  # negative sign to make positive value

            aug_timeout_nodes_reward = timeout_nodes_reward.reshape(aug_factor, batch_size, env.pomo_size)
            # shape: (augmentation, batch, pomo)
            max_pomo_timeout_nodes_reward, _ = aug_timeout_nodes_reward.max(dim=2)  # get best results from pomo
            no_aug_timeout_nodes_score = -max_pomo_timeout_nodes_reward[0, :].float()  # negative sign to make positive value
            max_aug_pomo_timeout_nodes_reward, _ = max_pomo_timeout_nodes_reward.max(dim=0)  # get best results from augmentation
            aug_timeout_nodes_score = -max_aug_pomo_timeout_nodes_reward.float()  # negative sign to make positive value
        else:
            dist = reward

        aug_reward = dist.reshape(aug_factor, int(env.batch_size/aug_factor) , env.pomo_size)

        if self.trainer_params["fsb_dist_only"]:
            infeasible = infeasible.reshape(aug_factor, int(env.batch_size/aug_factor), env.pomo_size)  # shape: (augmentation, batch, pomo)
            no_aug_feasible = (infeasible[0, :, :] == False).any(dim=-1)  # shape: (batch)
            aug_feasible = (infeasible == False).any(dim=0).any(dim=-1)  # shape: (batch)

            reward_masked = aug_reward.masked_fill(infeasible, -1e10) # get feasible results from pomo
            fsb_no_aug = reward_masked[0,:,:].max(dim=1, keepdim=True).values # shape: (augmentation, batch)
            fsb_aug = reward_masked.max(dim=0).values.max(dim=-1).values
            no_aug_score, aug_score = -fsb_no_aug, -fsb_aug

            infeasible_output = {
                "sol_infeasible_rate": infeasible.sum() / (env.batch_size), # * env.pomo_size * aug_factor),
                "ins_infeasible_rate": 1. - aug_feasible.sum() / batch_size,
                "no_aug_feasible": no_aug_feasible,
                "aug_feasible": aug_feasible
            }
        else:
            max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
            no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
            max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
            aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value
            infeasible_output = infeasible

        return no_aug_score, aug_score, infeasible_output, None, None

    def _val_and_stat(self, dir, val_path, env, batch_size=500, val_episodes=1000, compute_gap=False, epoch=1):
        no_aug_score_list, aug_score_list, no_aug_gap_list, aug_gap_list, sol_infeasible_rate_list, ins_infeasible_rate_list = [], [], [], [], [], []
        episode, no_aug_score, aug_score, sol_infeasible_rate, ins_infeasible_rate = 0, torch.zeros(0).to(self.device), torch.zeros(0).to(self.device), torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
        if self.trainer_params["fsb_dist_only"]:
            no_aug_feasible, aug_feasible = torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)

        while episode < val_episodes:
            remaining = val_episodes - episode
            bs = min(batch_size, remaining)
            data = env.load_dataset(os.path.join(dir, val_path), offset=episode, num_samples=bs)

            no_aug, aug, infsb_rate, pred_list, label_list  = self._val_one_batch(data, env, aug_factor=8, eval_type="argmax")
            if isinstance(aug, list):
                no_aug, no_aug_total_timeout, no_aug_timeout_nodes = no_aug
                aug, aug_total_timeout, aug_timeout_nodes = aug
            no_aug_score = torch.cat((no_aug_score, no_aug), dim=0)
            aug_score = torch.cat((aug_score, aug), dim=0)
            if isinstance(infsb_rate, dict):
                no_aug_fsb = infsb_rate["no_aug_feasible"]
                aug_fsb = infsb_rate["aug_feasible"]
                sol_infsb_rate = infsb_rate["sol_infeasible_rate"]
                ins_infsb_rate = infsb_rate["ins_infeasible_rate"]
                no_aug_feasible = torch.cat((no_aug_feasible, no_aug_fsb), dim=0)
                aug_feasible = torch.cat((aug_feasible, aug_fsb), dim=0)
            try:
                sol_infeasible_rate = torch.cat((sol_infeasible_rate, torch.tensor([sol_infsb_rate])), dim=0)
                ins_infeasible_rate = torch.cat((ins_infeasible_rate, torch.tensor([ins_infsb_rate])), dim=0)
            except:
                pass
            episode += bs

        if self.trainer_params["fsb_dist_only"]:
            print(">> Only feasible solutions are under consideration!")
            no_aug_score_list.append(round(no_aug_score[no_aug_feasible.bool()].mean().item(), 4))
            aug_score_list.append(round(aug_score[aug_feasible.bool()].mean().item(), 4))
        else:
            no_aug_score_list.append(round(no_aug_score.mean().item(), 4))
            aug_score_list.append(round(aug_score.mean().item(), 4))
        if sol_infeasible_rate.size(0) > 0:
            sol_infeasible_rate_list.append(round(sol_infeasible_rate.mean().item()*100, 3))
            ins_infeasible_rate_list.append(round(ins_infeasible_rate.mean().item() * 100, 3))

        try:
            sol_path = get_opt_sol_path(dir, env.problem, data[1].size(1), env.hardness, self.args.timewindows)
        except:
            sol_path = os.path.join(dir, "lkh_" + val_path)

        compute_gap = os.path.exists(sol_path)
        if compute_gap:
            opt_sol = load_dataset(sol_path, disable_print=True)[: val_episodes]
            if self.args.problem == "OPTWVP":
                grid_factor = -0.001
            opt_sol = torch.tensor([i[0]/grid_factor for i in opt_sol])
            if self.trainer_params["fsb_dist_only"]:
                gap = (no_aug_score[no_aug_feasible.bool()] - opt_sol[no_aug_feasible.bool()]) / opt_sol[no_aug_feasible.bool()] * 100
                aug_gap = (aug_score[aug_feasible.bool()] - opt_sol[aug_feasible.bool()]) / opt_sol[aug_feasible.bool()] * 100
            else:
                gap = (no_aug_score - opt_sol) / opt_sol * 100
                aug_gap = (aug_score - opt_sol) / opt_sol * 100
            no_aug_gap_list.append(round(gap.mean().item(), 4))
            aug_gap_list.append(round(aug_gap.mean().item(), 4))
            if epoch == 1 or epoch % 1 == 0:
                import pandas as pd
                df_gap = pd.DataFrame(aug_gap.cpu().numpy(), columns=["gap"])
                df_gap.to_csv(f'./data/{val_path}_gap_epoch{epoch}.csv', index=False)
            try:
                print(">> Val Score on {}: NO_AUG_Score: {}, NO_AUG_Gap: {}% --> AUG_Score: {}, AUG_Gap: {}%; Infeasible rate: {}% (solution-level), {}% (instance-level)".format(val_path, no_aug_score_list, no_aug_gap_list, aug_score_list, aug_gap_list, sol_infeasible_rate_list[0], ins_infeasible_rate_list[0]))
                return aug_score_list[0], aug_gap_list[0], [sol_infeasible_rate_list[0], ins_infeasible_rate_list[0]]
            except:
                print(">> Val Score on {}: NO_AUG_Score: {}, NO_AUG_Gap: {}% --> AUG_Score: {}, AUG_Gap: {}%".format(val_path, no_aug_score_list, no_aug_gap_list, aug_score_list, aug_gap_list))
                return aug_score_list[0], aug_gap_list[0], None

        else:
            print(">> Val Score on {}: NO_AUG_Score: {}, --> AUG_Score: {}; Infeasible rate: {}% (solution-level), {}% (instance-level)".format(val_path, no_aug_score_list, aug_score_list, sol_infeasible_rate_list[0], ins_infeasible_rate_list[0]))
            return aug_score_list[0], 0, [sol_infeasible_rate_list[0], ins_infeasible_rate_list[0]]
