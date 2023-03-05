#!/usr/bin/python

import os
import argparse
import numpy as np
from tqdm import trange
from datetime import datetime
import cv2
import pickle as pkl
import rospy

from agents import RandomShootingAgent, CEMAgent, MPPIAgent, DifferentialDriveAgent
from replay_buffer import ReplayBuffer
from logger import Logger
from train_utils import train_from_buffer
from utils import make_state_subscriber
from environment import Environment

# seed for reproducibility
# SEED = 0
# import torch; torch.manual_seed(SEED)
# np.random.seed(SEED)
SEED = np.random.randint(0, 1e9)


class Experiment:
    def __init__(self, robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, **params):
        self.params = params
        params["n_robots"] = len(self.robot_ids)

        if self.mpc_method == 'mppi':
            self.agent_class = MPPIAgent
        elif self.mpc_method == 'cem':
            self.agent_class = CEMAgent
        elif self.mpc_method == 'shooting':
            self.agent_class = RandomShootingAgent
        elif self.mpc_method == 'differential':
            self.agent_class = DifferentialDriveAgent
        else:
            raise ValueError

        self.agent = self.agent_class(params)
        self.env = Environment(robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, params)
        self.replay_buffer = ReplayBuffer(params, random=self.random_data)

        if self.mpc_method != 'differential':
            self.replay_buffer.restore(restore_path=self.restore_path)

        if self.sample_recent_buffer:
            self.replay_buffer_sample_fn = self.replay_buffer.sample_recent
        else:
            self.replay_buffer_sample_fn = self.replay_buffer.sample

        if self.load_agent:
            self.agent.restore()
        else:
            if self.pretrain_samples > 0:
                train_from_buffer(
                    self.agent, self.replay_buffer, validation_buffer=None,
                    pretrain_samples=self.pretrain_samples, save_agent=self.save_agent,
                    train_epochs=self.train_epochs, batch_size=self.batch_size,
                    meta=self.meta,
                )

        # self.logger = Logger(params)

        if self.exp_name is None:
            now = datetime.now()
            self.exp_name = now.strftime("%d_%m_%Y_%H_%M_%S")

        self.model_error_dir = os.path.expanduser(f"~/kamigami_data/model_errors/{self.exp_name}/")
        self.distance_cost_dir = os.path.expanduser(f"~/kamigami_data/distance_costs/{self.exp_name}/")
        self.plot_video_dir = os.path.expanduser(f"~/kamigami_data/plot_videos/{self.exp_name}/")
        self.real_video_dir = os.path.expanduser(f"~/kamigami_data/real_videos/{self.exp_name}/")
        self.params_pkl_dir = os.path.expanduser(f"~/kamigami_data/params_pkls/")

        dirs = [
            self.model_error_dir,
            self.distance_cost_dir,
            self.plot_video_dir,
            self.real_video_dir,
            self.params_pkl_dir,
        ]

        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        if self.eval_buffer_size != 0:
            self.start_eval = False
            self.replay_buffer.idx = self.pretrain_samples
        else:
            self.start_eval = True

        np.set_printoptions(suppress=True)

    def __getattr__(self, key):
        return self.params[key]

    def run(self):
        # warmup robot before running actual experiment
        if not self.debug:
            rospy.sleep(1)
            for _ in trange(5, desc="Warmup Steps"):
                random_max_action = np.random.choice([0.999, -0.999], size=2*self.n_robots)
                self.env.step(random_max_action)

        state = self.env.reset(self.agent, self.replay_buffer)
        # state = self.env.get_state()
        done = False
        episode = 0

        plot_imgs = []
        real_imgs = []

        model_errors = []
        distance_costs = []

        while not rospy.is_shutdown():
            if self.eval_buffer_size != 0 and self.replay_buffer.size >= self.eval_buffer_size and not self.start_eval:
                self.update_online = False
                self.start_eval = True
                episode = 0
                model_errors = []
                distance_costs = []
                plot_imgs = []
                real_imgs = []
                self.env.reverse_episode = True

                # self.agent = self.agent_class(self.params)
                # train_from_buffer(
                #     self.agent, self.replay_buffer, validation_buffer=None,
                #     pretrain_samples=self.eval_buffer_size, save_agent=self.save_agent,
                #     train_epochs=self.train_epochs, batch_size=self.batch_size,
                #     meta=self.meta,
                # )
                print("\n\nSTARTING EVAL\n")
                self.env.reset(self.agent, self.replay_buffer)
                continue

            t = rospy.get_time()
            goals = self.env.get_next_n_goals(self.agent.mpc_horizon)
            action, predicted_next_state = self.agent.get_action(state, goals)
            next_state, done = self.env.step(action)

            if state is not None and next_state is not None:
                self.replay_buffer.add(state, action, next_state)

                relevant_state = state[:2] if self.robot_goals else state[-3:-1]
                relevant_pred =  predicted_next_state[:2] if self.robot_goals else predicted_next_state[-3:-1]
                relevant_next = next_state[:2] if self.robot_goals else next_state[-3:-1]
                print(f"\nEPISODE STEP:", self.env.episode_step)
                print("DISTANCE FROM GOAL:", np.linalg.norm(relevant_next - goals[0, :2]))
                print("PREDICTION ERROR:", np.linalg.norm(relevant_pred - relevant_next))

            if self.update_online:
                for model in self.agent.models:
                    for _ in range(self.utd_ratio):
                        model.update(*self.replay_buffer_sample_fn(self.batch_size))

                        if self.replay_buffer.size == 20:
                            model.set_scalers(*self.replay_buffer.sample_recent(20))

            # log model errors and performance costs
            model_error = next_state - predicted_next_state
            distance_cost = np.linalg.norm(next_state[-3:-1] - goals[0, :2])

            model_errors.append(model_error)
            distance_costs.append(distance_cost)
            print("TIME:", rospy.get_time() - t, "s")

            if self.record_video:
                plot_img, real_img = self.env.render(done=done, episode=episode)
                plot_imgs.append(plot_img)
                real_imgs.append(real_img)

            if done:
                camera_imgs = np.array(self.env.camera_imgs)
                # np.save(f"/home/arovinsky/mar1_dump/real_video_ep{episode}.npy", camera_imgs)
                print("\n\nLOGGING VIDEO\n")
                log_video(camera_imgs, f"/home/arovinsky/mar1_dump/real_video_ep{episode}.avi", fps=30)

                model_error_fname = self.model_error_dir + f"episode{episode}.npy"
                distance_cost_fname = self.distance_cost_dir + f"episode{episode}.npy"

                model_error_arr = np.array(model_errors)
                distance_cost_arr = np.array(distance_costs)

                np.save(model_error_fname, model_error_arr)
                np.save(distance_cost_fname, distance_cost_arr)

                if self.use_object:
                    model_error_dist_norm = np.linalg.norm(model_error_arr[:, -3:-1], axis=1)
                else:
                    model_error_dist_norm = np.linalg.norm(model_error_arr[:, :2], axis=1)

                print(f"\nMODEL ERROR MEAN: {model_error_dist_norm.mean()} || STD: {model_error_dist_norm.std()}")
                print(f"DISTANCE ERROR MEAN: {distance_cost_arr.mean()} || STD: {distance_cost_arr.std()}")

                if self.record_video:
                    log_video(plot_imgs, self.plot_video_dir + f"plot_movie_{episode}.avi", fps=7)
                    log_video(real_imgs, self.real_video_dir + f"real_movie_{episode}.avi", fps=7)

                plot_imgs = []
                real_imgs = []

                episode += 1
                self.replay_buffer.dump()
                self.dump()

                if episode == self.n_episodes and self.start_eval:
                    rospy.signal_shutdown(f"Experiment finished! Did {self.n_episodes} rollouts.")
                    return

                for _ in trange(3, desc="Warmup Steps"):
                    random_max_action = np.random.choice([0.999, -0.999], size=2*self.n_robots)
                    self.env.step(random_max_action)

                state = self.env.reset(self.agent, self.replay_buffer)
            else:
                state = next_state

    def dump(self):
        self.params["buffer_save_path"] = self.replay_buffer.save_path
        self.params["buffer_restore_path"] = self.replay_buffer.restore_path

        with open(self.params_pkl_dir + f"{self.exp_name}.pkl", "wb") as f:
            pkl.dump(self.params, f)


def log_video(imgs, filepath, fps=7):
    height, width = imgs[0].shape[0], imgs[0].shape[1]
    video = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'XVID'), fps, (width,height))

    for img in imgs:
        video.write(img)

    video.release()

def main(args):
    print("INITIALIZING NODE")
    rospy.init_node("run_experiment")

    robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, tf_buffer, tf_listener = make_state_subscriber(args.robot_ids)
    experiment = Experiment(robot_pos, object_pos, corner_pos, robot_vel, object_vel, action_timestamp, **vars(args))
    experiment.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-mpc_method', type=str, default='mppi')
    parser.add_argument('-trajectory', type=str, default='8')
    parser.add_argument('-restore_path', type=str, default=None)

    parser.add_argument('-alpha', type=float, default=0.8)
    parser.add_argument('-n_best', type=int, default=30)
    parser.add_argument('-refine_iters', type=int, default=5)

    parser.add_argument('-gamma', type=int, default=50)
    parser.add_argument('-beta', type=float, default=0.5)
    parser.add_argument('-noise_std', type=float, default=2)

    # generic
    parser.add_argument('-robot_ids', nargs='+', type=int, default=[0])
    parser.add_argument('-object_id', type=int, default=3)
    parser.add_argument('-use_object', action='store_true')
    parser.add_argument('-exp_name', type=str, default=None)

    parser.add_argument('-n_episodes', type=int, default=3)
    parser.add_argument('-tolerance', type=float, default=0.04)
    parser.add_argument('-episode_length', type=int, default=150)
    parser.add_argument('-eval_buffer_size', type=int, default=0)

    parser.add_argument('-meta', action='store_true')
    parser.add_argument('-pretrain_samples', type=int, default=500)
    parser.add_argument('-train_epochs', type=int, default=200)

    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-save_agent', action='store_true')
    parser.add_argument('-load_agent', action='store_true')
    parser.add_argument('-record_video', action='store_true')
    parser.add_argument('-random_data', action='store_true')

    # agent
    parser.add_argument('-ensemble_size', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=10000)
    parser.add_argument('-update_online', action='store_true')
    parser.add_argument('-sample_recent_buffer', action='store_true')
    parser.add_argument('-utd_ratio', type=int, default=3)

    parser.add_argument('-mpc_horizon', type=int, default=5)
    parser.add_argument('-mpc_samples', type=int, default=200)
    parser.add_argument('-robot_goals', action='store_true')

    # model
    parser.add_argument('-hidden_dim', type=int, default=200)
    parser.add_argument('-hidden_depth', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.001)

    parser.add_argument('-scale', action='store_true')
    parser.add_argument('-dist', action='store_true')
    parser.add_argument('-std', type=bool, default=0.01)

    # replay buffer
    parser.add_argument('-save_freq', type=int, default=50) # TODO implement this
    parser.add_argument('-buffer_capacity', type=int, default=10000)

    args = parser.parse_args()
    main(args)

"""
no ensemble, hidden_depth=1, gamma=30, discount=0.8, horizon=5
DISTANCE ERROR MEAN: 0.08660430441722838 || STD: 0.08202011627116493

no ensemble, hidden_depth=2, gamma=30, discount=0.8, horizon=5
DISTANCE ERROR MEAN: 0.08008191716750115 || STD: 0.06520245115634518

ensemble=3, hidden_depth=1, gamma=30, discount=0.8, horizon=5
DISTANCE ERROR MEAN: 0.08913504793041617 || STD: 0.099393394428367

ensemble=3, hidden_depth=2, gamma=30, discount=0.8, horizon=5
DISTANCE ERROR MEAN: 0.07581990710792152 || STD: 0.07214221137403418

ensemble=3, hidden_depth=2, gamma=50, discount=0.8, horizon=5
DISTANCE ERROR MEAN: 0.06275876273060346 || STD: 0.057002978120098566

ensemble=3, hidden_depth=2, gamma=70, discount=0.8, horizon=5
DISTANCE ERROR MEAN: 0.05226356916374783 || STD: 0.0433097831055788

ensemble=3, hidden_depth=2, gamma=90, discount=0.8, horizon=5
DISTANCE ERROR MEAN: 0.0610607449694273 || STD: 0.065733525047127

ensemble=3, hidden_depth=2, gamma=70, discount=0.9, horizon=5
DISTANCE ERROR MEAN: 0.04783398583506517 || STD: 0.03658398860961526

ensemble=3, hidden_depth=2, gamma=70, discount=1, horizon=5
DISTANCE ERROR MEAN: 0.07302833795775061 || STD: 0.06251048779332265

ensemble=3, hidden_depth=2, gamma=70, discount=0.9, horizon=7
DISTANCE ERROR MEAN: 0.04889878846064686 || STD: 0.027612082797363848

ensemble=3, hidden_depth=2, gamma=70, discount=0.85, horizon=7
DISTANCE ERROR MEAN: 0.06159934594987193 || STD: 0.03707809897572567

ensemble=3, hidden_depth=2, gamma=70, discount=0.95, horizon=7
DISTANCE ERROR MEAN: 0.06697552109165145 || STD: 0.05679994352659188

ensemble=3, hidden_depth=2, gamma=70, discount=0.9, horizon=7 (verification run)
DISTANCE ERROR MEAN: 0.0827479175162651 || STD: 0.07333474529238258     (robots low battery)


2 robots, 40 steps
DISTANCE ERROR MEAN: 0.08681493377911005 || STD: 0.07678928493809394

3 robots, 40 steps
DISTANCE ERROR MEAN: 0.13480097606944455 || STD: 0.11659018068372139

3 robots, 50 steps
DISTANCE ERROR MEAN: 0.11397670947219868 || STD: 0.08400565406814478

### 3 robots ###
500 data
DISTANCE ERROR MEAN: 0.11321832694473398 || STD: 0.10823851087263592

pretrain steps 500
DISTANCE ERROR MEAN: 0.1452882222296189 || STD: 0.11353098590115421

batch 100
DISTANCE ERROR MEAN: 0.10494090510645028 || STD: 0.09824747745212195

gamma 50
DISTANCE ERROR MEAN: 0.13276119591727087 || STD: 0.131304235616815

gamma 90
DISTANCE ERROR MEAN: 0.10160475800074537 || STD: 0.07468809875436884

no ensemble
DISTANCE ERROR MEAN: 0.1798249048984371 || STD: 0.16621333287050097

ensemble 5
DISTANCE ERROR MEAN: 0.1519015035615877 || STD: 0.12349650763254517

--------
S
30 online
DISTANCE ERROR MEAN: 0.1453261451592454 || STD: 0.14809046036980614

40 online
DISTANCE ERROR MEAN: 0.08513573009861082 || STD: 0.06027042066525054
DISTANCE ERROR MEAN: 0.10753619857756619 || STD: 0.08289441516271857
DISTANCE ERROR MEAN: 0.06061867211918317 || STD: 0.04324651454675491

50 online
DISTANCE ERROR MEAN: 0.03781279692971551 || STD: 0.027158208047561416
DISTANCE ERROR MEAN: 0.06576219982423469 || STD: 0.05604051942245982

60 online
DISTANCE ERROR MEAN: 0.04545153791760536 || STD: 0.030865017683060422
DISTANCE ERROR MEAN: 0.050151552243281154 || STD: 0.048407849671707436
DISTANCE ERROR MEAN: 0.042866446340227994 || STD: 0.026053458796755653


30 offline
DISTANCE ERROR MEAN: 0.12132439498150954 || STD: 0.07941562406785879

40 offline
DISTANCE ERROR MEAN: 0.1291909589236959 || STD: 0.11153518400939245

50 offline
DISTANCE ERROR MEAN: 0.08660938408932575 || STD: 0.08294878534214319

60 offline
DISTANCE ERROR MEAN: 0.05074773490812932 || STD: 0.032017062436823764

80 offline
DISTANCE ERROR MEAN: 0.043375499045631684 || STD: 0.03759569380864598

100 offline
DISTANCE ERROR MEAN: 0.03543915114038648 || STD: 0.022265690480709508


W
30 online
DISTANCE ERROR MEAN: 0.12364659675589902 || STD: 0.09210150078788215

40 steps online
DISTANCE ERROR MEAN: 0.06546685887549143 || STD: 0.04290647167606537

50 steps online
DISTANCE ERROR MEAN: 0.07858967471583123 || STD: 0.0787754805013675
DISTANCE ERROR MEAN: 0.08334296393754932 || STD: 0.054462092147830765
DISTANCE ERROR MEAN: 0.06264222275776812 || STD: 0.04802377555000213

60 steps online
DISTANCE ERROR MEAN: 0.05178441402397384 || STD: 0.04228615109603794

80 steps online
DISTANCE ERROR MEAN: 0.04363296753908305 || STD: 0.02963506076459486

100 steps online
DISTANCE ERROR MEAN: 0.042789842192290733 || STD: 0.03241297118476115
DISTANCE ERROR MEAN: 0.04045665074171182 || STD: 0.028786505428695774


30 offline
DISTANCE ERROR MEAN: 0.12392937588062199 || STD: 0.09702191064027618

40 offline
DISTANCE ERROR MEAN: 0.1221309871808771 || STD: 0.10556864905801173

50 offline
DISTANCE ERROR MEAN: 0.07795111421502011 || STD: 0.05349630386346188

60 offline
DISTANCE ERROR MEAN: 0.05196163844152554 || STD: 0.03589218555582435

80 offline
DISTANCE ERROR MEAN: 0.047337856389672577 || STD: 0.035801792845641896

100 offline
DISTANCE ERROR MEAN: 0.03619928478549898 || STD: 0.023948265898312594

---------------------------------------
budget grid

200 budget
offline 0, online 200
DISTANCE ERROR MEAN: 0.05900206540990112 || STD: 0.03697986653742203

offline 50, online 150
DISTANCE ERROR MEAN: 0.11757103718610445 || STD: 0.11732155506273652
-------
DISTANCE ERROR MEAN: 0.055704950028909996 || STD: 0.03745416578697313

0.063

offline 100, online 100
DISTANCE ERROR MEAN: 0.0773383166899225 || STD: 0.05075649263713999

offline 150, online 50
DISTANCE ERROR MEAN: 0.058581546166214554 || STD: 0.03935480932502959

offline 200, online 0
DISTANCE ERROR MEAN: 0.0815464171288694 || STD: 0.07098279585704972



500 budget
offline 0, online 500
DISTANCE ERROR MEAN: 0.13919046550265027 || STD: 0.12838400649697776

offline 125, online 375
DISTANCE ERROR MEAN: 0.05916047953116735 || STD: 0.048468081423366526

offline 250, online 250
DISTANCE ERROR MEAN: 0.04726254420454711 || STD: 0.03792566678009408
--
DISTANCE ERROR MEAN: 0.04704502383521745 || STD: 0.032775743590780046
DISTANCE ERROR MEAN: 0.05362031455374961 || STD: 0.04160427871623092

offline 375, online 125
DISTANCE ERROR MEAN: 0.07734169575187776 || STD: 0.06415108012895403

offline 500, online 0
DISTANCE ERROR MEAN: 0.05531861076242347 || STD: 0.0363525361312538
DISTANCE ERROR MEAN: 0.062049522049403026
DISTANCE ERROR MEAN: 0.06081894427562993 || STD: 0.05190957908230576
DISTANCE ERROR MEAN: 0.057623483738053975 || STD: 0.03420731826572024
first episode: DISTANCE ERROR MEAN: 0.03507091795975834 || STD: 0.021934623109792915
DISTANCE ERROR MEAN: 0.04242788845328144 || STD: 0.02936095877448667




400 budget
offline 0, online 400
DISTANCE ERROR MEAN: 0.1199551511422212 || STD: 0.11916185834968426

# REDO THIS ONE
offline 100, online 300
DISTANCE ERROR MEAN: 0.052394407560618014 || STD: 0.03640035032980452

offline 200, online 200
DISTANCE ERROR MEAN: 0.047444738970924154 || STD: 0.02747415133386499

offline 300, online 100
DISTANCE ERROR MEAN: 0.0579549914979674 || STD: 0.042670597398293486

offline 400, online 0
DISTANCE ERROR MEAN: 0.051359848329762404 || STD: 0.03314023575828148
----
MODEL ERROR MEAN: 0.028064553988406873 || STD: 0.015994339433717617
DISTANCE ERROR MEAN: 0.046523936697114375 || STD: 0.032229946329893275


# 600 budget
# offline 0, online 600

# offline 150, online 450
# #this was the one the robot disconnected during

# offline 300, online 300
# DISTANCE ERROR MEAN: 0.05353105279031655 || STD: 0.03179030987689723
# DISTANCE ERROR MEAN: 0.06188439533220234 || STD: 0.053278693062311505
# DISTANCE ERROR MEAN: 0.05722177152962711 || STD: 0.05202330946837465
# DISTANCE ERROR MEAN: 0.054686673041148964 || STD: 0.03906939159038083
# DISTANCE ERROR MEAN: 0.07460215716923423 || STD: 0.06158442753982853

# offline 450, online 150
# DISTANCE ERROR MEAN: 0.07839866534389242 || STD: 0.0869548273912581

# offline 600, online 0



300 budget
offline 0, online 300
DISTANCE ERROR MEAN: 0.10573706183710509 || STD: 0.07382054714726942

offline 75, online 225
DISTANCE ERROR MEAN: 0.0493623571178731 || STD: 0.03274392232398357

offline 150, online 150
# DISTANCE ERROR MEAN: 0.08642308810061465 || STD: 0.0630064443761231
# DISTANCE ERROR MEAN: 0.06356004234168174 || STD: 0.052974981640148
# DISTANCE ERROR MEAN: 0.0565491164576109 || STD: 0.045504967508146185
# DISTANCE ERROR MEAN: 0.07763467088310368 || STD: 0.05915834413828475
# DISTANCE ERROR MEAN: 0.07264674293621905 || STD: 0.07327767403604207
DISTANCE ERROR MEAN: 0.05751330620204849 || STD: 0.05015139775936857

offline 225, online 75
# DISTANCE ERROR MEAN: 0.0650584140000112 || STD: 0.044708402088415006
DISTANCE ERROR MEAN: 0.073822112229081 || STD: 0.06444292373158414

offline 300, online 0
# DISTANCE ERROR MEAN: 0.09180349154575881 || STD: 0.06143724124428051
DISTANCE ERROR MEAN: 0.049053515176116974 || STD: 0.027830896999083345
DISTANCE ERROR MEAN: 0.09629106288129723 || STD: 0.09713499548053066
DISTANCE ERROR MEAN: 0.06496860767111963 || STD: 0.0555608970549477

----------------------------

3 ROBOTS ONLINE, S

40
DISTANCE ERROR MEAN: 0.12882051489966936 || STD: 0.10871160634174419

50
DISTANCE ERROR MEAN: 0.08788492145506244 || STD: 0.09627631822543747

60
DISTANCE ERROR MEAN: 0.060560546217134914 || STD: 0.04883899330758632

80
DISTANCE ERROR MEAN: 0.04749598667477746 || STD: 0.047906536860510336

100
DISTANCE ERROR MEAN: 0.03765142676959593 || STD: 0.031380228676241476



1 ROBOT ONLINE, S

40
DISTANCE ERROR MEAN: 0.1241700460256072 || STD: 0.09872223154292022

50
DISTANCE ERROR MEAN: 0.07984592184926866 || STD: 0.06511776327505864

60
DISTANCE ERROR MEAN: 0.04873282880043172 || STD: 0.033689535493317165

80
DISTANCE ERROR MEAN: 0.04092268207552183 || STD: 0.036004644397826274

100
DISTANCE ERROR MEAN: 0.03463663442188167 || STD: 0.024135032232959062


rosrun ros_stuff run_experiment.py -mpc_method=mppi -gamma=80 -n_episodes=5 -tolerance=0.04 -episode_length=50 -train_epochs=300 -save_agent -ensemble_size=1 -batch_size=1000 -utd_ratio=3 -mpc_horizon=7 -mpc_samples=300 -hidden_dim=200 -hidden_depth=2 -lr=0.001 -scale -std=0.02 -buffer_capacity=10000 -robot_ids 0 2 -use_object -trajectory=S -record_video -beta=0.9 -noise_std=2 -dist -random_data -update_online -eval_buffer_size=600 -exp_name=2object_budget400_online300_S50 -pretrain_samples=300




rosrun ros_stuff collect_data.py -buffer_capacity=10000 -robot_ids 0 1 2 -use_object -random_data -n_samples=500 -beta -exp_name=3object_beta500_2
rosrun ros_stuff run_experiment.py -mpc_method=mppi -gamma=80 -n_episodes=5 -tolerance=0.04 -episode_length=100 -train_epochs=300 -save_agent -ensemble_size=1 -batch_size=1000 -utd_ratio=3 -mpc_horizon=7 -mpc_samples=300 -hidden_dim=200 -hidden_depth=2 -lr=0.001 -scale -std=0.02 -buffer_capacity=10000 -robot_ids 0 1 2 -use_object -trajectory=S -record_video -beta=0.9 -noise_std=2 -dist -random_data -update_online -exp_name=3object_beta500_online_S100 -pretrain_samples=500

"""