
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.envs.grid_world_env import GridWorldEnv

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy

import gym
import pickle
import os
import joblib
import numpy as np
import os
import tensorflow as tf
import unittest

from hgail.critic.critic import WassersteinCritic
from hgail.misc.datasets import CriticDataset, RecognitionDataset
from hgail.envs.envs import TwoRoundNondeterministicRewardEnv
from hgail.policies.categorical_latent_var_mlp_policy import CategoricalLatentVarMLPPolicy
from hgail.algos.gail import GAIL
from hgail.policies.latent_sampler import UniformlyRandomLatentSampler
from hgail.core.models import CriticNetwork, ObservationActionMLP
from hgail.recognition.recognition_model import RecognitionModel
from hgail.policies.scheduling import ConstantIntervalScheduler
from hgail.misc.utils import RewardHandler

def train_gail(
        session, 
        env, 
        dataset,
        obs_dim=1,
        act_dim=2,
        n_itr=20,
        use_env_rewards=False,
        discount=.99,
        batch_size=4000,
        critic_scale=1.,
        gail_step_size=.01,
        critic_learning_rate=.001,
        policy_hid_layer_dims=[32,32],
        gradient_penalty=.1,
        critic_n_train_epochs=1,
        sampler_args=dict(),
        return_algo=False):

    network = CriticNetwork(hidden_layer_dims=[32,32])
    critic = WassersteinCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        dataset=dataset, 
        network=network,
        verbose=2,
        gradient_penalty=gradient_penalty,
        optimizer=tf.train.AdamOptimizer(critic_learning_rate, beta1=.5, beta2=.9),
        n_train_epochs=critic_n_train_epochs
    )
    policy = CategoricalMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=policy_hid_layer_dims
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)  # ZeroBaseline(env_spec=env.spec)

    reward_handler = RewardHandler(
        use_env_rewards=use_env_rewards,
        critic_final_scale=critic_scale)

    algo = GAIL(
        critic=critic,
        reward_handler=reward_handler,
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=500,
        n_itr=n_itr,
        discount=discount,
        step_size=gail_step_size,
        sampler_args=sampler_args,
    )
    session.run(tf.global_variables_initializer())

    if return_algo:
        return algo

    algo.train(sess=session)

    return policy, critic


def onlinesession_train_gail(
        session,
        env,
        critic,
        policy,
        obs_dim=1,
        act_dim=2,
        n_itr=20,
        use_env_rewards=False,
        discount=.99,
        batch_size=4000,
        critic_scale=1.,
        gail_step_size=.01,
        critic_learning_rate=.001,
        policy_hid_layer_dims=[32,32],
        gradient_penalty=.1,
        critic_n_train_epochs=1,
        sampler_args=dict(),
        return_algo=False):
    # dataset for critic changes and params should start from where they left

    baseline = LinearFeatureBaseline(env_spec=env.spec)  # ZeroBaseline(env_spec=env.spec)

    reward_handler = RewardHandler(
        use_env_rewards=use_env_rewards,
        critic_final_scale=critic_scale)

    algo = GAIL(
        critic=critic,
        reward_handler=reward_handler,
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=500,
        n_itr=n_itr,
        discount=discount,
        step_size=gail_step_size,
        sampler_args=sampler_args,
    )
    session.run(tf.global_variables_initializer())

    if return_algo:
        return algo

    algo.train(sess=session)

    return policy, critic


class TestGAIL(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.set_random_seed(1)
        np.random.seed(1)
        tf.reset_default_graph()    

    # def test_gail_one_round_deterministic_env(self):
    #
    #     with tf.Session() as session:
    #
    #         n_expert_samples = 1000
    #         rx = np.ones((n_expert_samples, 1))
    #         ra = np.zeros((n_expert_samples, 2))
    #         ra[:,1] = 1 # one hot actions
    #         dataset = CriticDataset(dict(observations=rx, actions=ra), batch_size=1000)
    #
    #         env = TfEnv(GymEnv("OneRoundDeterministicReward-v0", force_reset=True))
    #
    #         policy, critic = train_gail(session, env, dataset, use_env_rewards=False, n_itr=20)
    #         dist = policy.dist_info([[1.]])['prob']
    #         np.testing.assert_array_almost_equal(dist, [[0,1]], 2)
    #
    # def test_gail_two_round_deterministic_env(self):
    #
    #     with tf.Session() as session:
    #
    #         # dataset of one-hot obs and acts
    #         # optimal actions: 0, 1
    #         # first state
    #         n_expert_samples = 1000
    #         half = int(n_expert_samples / 2)
    #         rx = np.zeros((n_expert_samples, 3))
    #         rx[:half,2] = 1
    #         rx[half:,0] = 1
    #         ra = np.zeros((n_expert_samples, 2))
    #         ra[:half,0] = 1
    #         ra[half:,1] = 1
    #         dataset = CriticDataset(dict(observations=rx, actions=ra), batch_size=1000)
    #
    #         env = TfEnv(GymEnv("TwoRoundDeterministicReward-v0", force_reset=True))
    #
    #         policy, critic = train_gail(
    #             session,
    #             env,
    #             dataset,
    #             obs_dim=3,
    #             act_dim=2,
    #             use_env_rewar    def reset(self):
    #             critic_scale=1.,
    #             n_itr=15,
    #             policy_hid_layer_dims=[32,32],
    #             batch_size=4000,
    #             critic_learning_rate=.001,
    #             gradient_penalty=1.,
    #             critic_n_train_epochs=10
    #         )
    #         dist_2 = policy.dist_info([[0.,0.,1.]])['prob']
    #         dist_0 = policy.dist_info([[1.,0.,0.]])['prob']
    #         np.testing.assert_array_almost_equal(dist_2, [[1,0]], 1)
    #         np.testing.assert_array_almost_equal(dist_0, [[0,1]], 1)

    # def test_gail_two_round_stochastic_env(self):
    #
    #     with tf.Session() as session:
    #
    #         # dataset of one-hot obs and acts
    #         # optimal actions: 0, 1
    #         # first state
    #         n_expert_samples = 10#00
    #         half = int(n_expert_samples / 2)
    #         rx = np.zeros((n_expert_samples, 3))
    #         rx[:half,2] = 1
    #         rx[half:,0] = 1
    #         ra = np.zeros((n_expert_samples, 2))
    #         ra[:half,0] = 1
    #         ra[half:,1] = 1
    #         dataset = CriticDataset(dict(observations=rx, actions=ra), batch_size=10)#00)
    #
    #         env = TfEnv(TwoRoundNondeterministicRewardEnv())
    #
    #         policy, critic = train_gail(
    #             session,
    #             env,
    #             dataset,
    #             obs_dim=3,
    #             act_dim=2,
    #             use_env_rewards=False,
    #             critic_scale=1.,
    #             n_itr=15,
    #             policy_hid_layer_dims=[32,32],
    #             batch_size=40,#00,
    #             critic_learning_rate=.001,
    #             gradient_penalty=1.,
    #             critic_n_train_epochs=10,
    #             sampler_args=dict(n_envs=10)
    #         )
    #         dist_2 = policy.dist_info([[0.,0.,1.]])['prob']
    #         dist_0 = policy.dist_info([[1.,0.,0.]])['prob']
    #         np.testing.assert_array_almost_equal(dist_2, [[1,0]], 1)
    #         np.testing.assert_array_almost_equal(dist_0, [[0,1]], 1)

    def get_home(self):
        return "/media/saurabharora/f281cd1b-fbb4-40ff-a2b0-78a56619e39d/saurabharora"

    def test_gail_GridWorldEnv(self):

        f = open(self.get_home() + "/hgail/patrol_data_lba.csv", "w")
        f.write("")
        f.close()

        with tf.Session() as session:

            n_expert_samples = 5
            self.rllab_env = GridWorldEnv("Patrol")  # "chain")
            critic_dataset_batch_size = 1000
            network_hidden_sizes = [64,8]
            critic_learning_rate = 0.01  #.001
            critic_gradient_penalty = 0.5  #1
            critic_n_train_epochs = 5
            gail_batch_size = 1500
            gail_step_size = 0.005
            self.no_of_trials = 20
            self.no_of_sessions = 20
            self.bagging_iterations = 1
            # 2 d array for storing lba value after each session
            self.lba = np.zeros((self.no_of_sessions, self.no_of_trials))

            # self.rllab_env = GridWorldEnv("8x8")
            res = self.rllab_env.reset()

            # expert's policy

            os.environ['THEANO_FLAGS'] = 'device=cpu,mode=FAST_COMPILE,optimizer=None'

            env = TfEnv(self.rllab_env)


            res = self.rllab_env.reset()

            for i in range(0, 120):
                state = self.rllab_env.state
                if self.rllab_env.desc_str == "Patrol":
                    state = self.rllab_env.state_enum[state]
                    x = (state // 4) // self.rllab_env.n_col
                    y = (state // 4) % self.rllab_env.n_col
                    th = state % 4
                    print("state:", x, y, th)
                else:
                    x = state // self.rllab_env.n_col
                    y = state % self.rllab_env.n_col
                    print("state:", x, y)

                # rx[i, self.rllab_env.state] = 1
                a = np.random.choice([0,1,2,3])
                # a = policy_expert.get_action(self.rllab_env.state)[0]
                print("action:", a)
                # ra[i, a] = 1
                res = self.rllab_env.step(a)

            # print(policy.get_params())
            # print(policy.get_param_values())
            #
            # exit()

            str_touploadpath = self.get_home()+"/NIPscode_perimeterpatrolcode_May29/patrolstudy/toupload/"
            filecontents = []
            f = open(str_touploadpath+"recd_convertedstates_14_14_300states_filtered.log", "r")
            for line in f:
                loc_act = line.strip().split(",") #string.split(line.strip(), ",")
                if loc_act[0] == "ENDREC":
                    break
                elif loc_act[0] == "None":
                    pass  # change GAIL to receive missing timesteps
                else:
                    (x, y, th) = (int(loc_act[0]), int(loc_act[1]), int(loc_act[2]))
                    state_value = (x * self.rllab_env.n_col + y)*4+ th
                    state = list(self.rllab_env.state_enum.keys())\
                        [list(self.rllab_env.state_enum.values()).index(state_value)]

                    if loc_act[3] == "PatrolActionMoveForward":
                        action = 1
                    elif loc_act[3] == "PatrolActionTurnLeft":
                        action = 0
                    elif loc_act[3] == "PatrolActionTurnRight":
                        action = 2
                    elif loc_act[3] == "PatrolActionStop":
                        action = 3
                filecontents.append((state, action))

            # print(filecontents)

            f.close()

            # exit()
            # Read policy
            correctboydpolicy = self.get_home() + \
                                "/NIPscode_perimeterpatrolcode_May29/patrolstudy/boydpolicy_mdppatrolcontrol_bkup"
            f = open(correctboydpolicy, "r")

            self.boydpolicy = {}
            for stateaction in f:
                temp = stateaction.strip().split(" = ")
                if len(temp) < 2: continue
                state = temp[0]
                action = temp[1]

                if action == "MoveForwardAction":
                    action = 1
                elif action == "TurnLeftAction":
                    action = 0
                elif action == "TurnRightAction":
                    action = 2
                elif action == "StopAction":
                    action = 3

                pstate = state[1: len(state) - 1]
                pieces = pstate.split(",")
                ps = (int(pieces[0]) * self.rllab_env.n_col + int(pieces[1])) * 4 + int(pieces[2])
                if ps in list(self.rllab_env.state_enum.values()):
                    state = list(self.rllab_env.state_enum.keys()) \
                        [list(self.rllab_env.state_enum.values()).index(ps)]

                    self.boydpolicy[state] = action

            f.close()

            # To start with random critic policy values,
            # train critic using empty dataset n_itr=1, batch=10

            if self.rllab_env.desc_str == "Patrol":
                rx = np.zeros((n_expert_samples, len(self.rllab_env.state_enum)))
            else:
                rx = np.zeros((n_expert_samples, self.rllab_env.n_row * self.rllab_env.n_col))
            ra = np.zeros((n_expert_samples, 4))

            st_dataset = CriticDataset(dict(observations=rx, actions=ra), batch_size=critic_dataset_batch_size)
            network = CriticNetwork(hidden_layer_dims=network_hidden_sizes)

            critic = WassersteinCritic(
                obs_dim=len(self.rllab_env.state_enum),
                act_dim=4,
                dataset=st_dataset,
                network=network,
                # verbose=2,
                # gradient_penalty=critic_gradient_penalty,
                optimizer=tf.train.AdamOptimizer(),  # critic_learning_rate, beta1=.5, beta2=.9),
                n_train_epochs=critic_n_train_epochs
            )

            from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
            self.policy = CategoricalMLPPolicy(
                name="policy",
                env_spec=env.spec,
                hidden_sizes=network_hidden_sizes
            )

            baseline = LinearFeatureBaseline(env_spec=env.spec)  # ZeroBaseline(env_spec=env.spec)

            reward_handler = RewardHandler(
                use_env_rewards=False,
                # critic_final_scale=1.
            )

            algo = GAIL(
                critic=critic,
                reward_handler=reward_handler,
                env=env,
                policy=self.policy,
                baseline=baseline,
                # batch_size=10,
                # max_path_length=100,
                n_itr=1,
                # discount=.99,
                # step_size=gail_step_size,
                # sampler_args=dict(),
            )

            session.run(tf.global_variables_initializer())
            algo.train(sess=session)


            st_critic_values = critic.network.get_param_values()
            st_policy_values = self.policy.get_param_values()
            algo.n_itr = 250
            # algo.batch_size = 5000  # gail_batch_size

            # print(critic_values)
            # print(policy_values)

            f = open(self.get_home() + "/hgail/patrol_data_lba.csv", "a")

            for trial_no in range(0, self.no_of_trials):

                f.write("\n")
                # reset values at the beginning of every trial
                critic_values = st_critic_values
                policy_values = st_policy_values
                critic.dataset = st_dataset

                for sess_no in range(0, self.no_of_sessions):  # len(filecontents) // n_expert_samples):

                    # restarting for every session
                    if self.rllab_env.desc_str == "Patrol":
                        rx = np.zeros((self.bagging_iterations*n_expert_samples, len(self.rllab_env.state_enum)))
                    else:
                        rx = np.zeros((self.bagging_iterations*n_expert_samples, self.rllab_env.n_row * self.rllab_env.n_col))
                    ra = np.zeros((self.bagging_iterations*n_expert_samples, 4))

                    for i in range(0 + sess_no * n_expert_samples, (sess_no + 1) * n_expert_samples):

                        state = filecontents[i][0]
                        # state = (i-sess_no*n_expert_samples) % len(self.boydpolicy)
                        state_value = self.rllab_env.state_enum[state]
                        x = (state_value // 4) // self.rllab_env.n_col
                        y = (state_value // 4) % self.rllab_env.n_col
                        th = state_value % 4
                        # print("state:", x, y, th)
                        # if x<=18: #for 100% visibility
                        # if x<=14: #for 73% visibility
                        if x==0 and y<=4: # for 30% visibility
                            state = list(self.rllab_env.state_enum.keys()) \
                                [list(self.rllab_env.state_enum.values()).index(state_value)]
                            rx[i-sess_no*n_expert_samples, state] = 1
                            a = filecontents[i][1]
                            # a = self.boydpolicy[state]
                            # print("action:", a)
                            ra[i-sess_no*n_expert_samples, a] = 1

                    for bag in range(1,self.bagging_iterations):
                        rx[bag*n_expert_samples:(bag+1)*n_expert_samples, :] = rx[0:n_expert_samples,:]
                        ra[bag*n_expert_samples:(bag+1)*n_expert_samples, :] = ra[0:n_expert_samples,:]

                    # exit()
                    dataset = CriticDataset(dict(observations=rx, actions=ra), batch_size=critic_dataset_batch_size)
                    critic.dataset = dataset
                    critic.network.set_param_values(critic_values)
                    self.policy.set_param_values(policy_values)
                    session.run(tf.global_variables_initializer())
                    algo.train(sess=session)
                    critic_values = critic.network.get_param_values()
                    policy_values = self.policy.get_param_values()
                    res = self.rllab_env.reset()

                    for i in range(0, 300):
                        state = self.rllab_env.state
                        state_value = self.rllab_env.state_enum[state]
                        x = (state_value // 4) // self.rllab_env.n_col
                        y = (state_value // 4) % self.rllab_env.n_col
                        th = state_value % 4
                        # print("state:", x, y, th)
                        state = list(self.rllab_env.state_enum.keys()) \
                            [list(self.rllab_env.state_enum.values()).index(state_value)]
                        print("state:", x, y, th)
                        a = self.policy.get_action(state)[0]
                        print("action:", a)
                        res = self.rllab_env.step(a)

                    self.compute_store_LBA(sess_no, trial_no)
                    f.write(str(sess_no)+":"+str(self.lba[sess_no, trial_no]) + ",")

            print(np.average(self.lba, axis=1))

            # for trial_no in range(0,self.no_of_trials):
                # for sess_no in range(0,self.no_of_sessions):
            f.close()

            import datetime
            import smtplib
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login("sonu.1987.arora@gmail.com", "s_arora1987")
            msgsubject = "Notification - GAIL Simulation Finished \n"
            msg = "Sent at : "
            msg += str(datetime.datetime.now()) + "\n"
            msg += "Simulation Finished"
            message = 'Subject: {}\n\n{}'.format(msgsubject, msg)
            print("sending Notification")
            server.sendmail("sonu.1987.arora@gmail.com", "sarora@udel.edu", message)
            server.quit()

    def compute_store_LBA(self, sess_no, trial_no):


        policyaccuracy = []
        lba = 0.0
        # i = 0
        # self.patPolicies = [policy]
        #
        # for patroller in self.patPolicies:
        totalsuccess = 0
        totalstates = 0

        # print(self.boydpolicy.keys())
        # exit()

        prob_lw_bnd = 0.8

        import operator

        for state in list(self.rllab_env.state_enum.keys()):
            if state in list(self.boydpolicy.keys()):  # check key existence
                totalstates += 1

                state_value = self.rllab_env.state_enum[state]
                x = (state_value // 4) // self.rllab_env.n_col
                y = (state_value // 4) % self.rllab_env.n_col
                th = state_value % 4

                res = self.policy.get_action(state)
                index, value = max(enumerate(res[1]["prob"]), key=operator.itemgetter(1))
                print(index, value)

                # if res[1]["prob"][action] < prob_lw_bnd:
                #     print("found action with prob < "+str(prob_lw_bnd)+\
                #           " for state"+str((x,y,th)))
                    # exit()
                # action = res[0]

                if (self.boydpolicy[state] == index):
                    # print("found a matching action for patroller "+str(i))
                    totalsuccess += 1

            # print("totalstates, totalsuccess: "+str(totalstates)+", "+str(totalsuccess))
            if float(totalstates) == 0:
                break

        lba = float(totalsuccess) / float(totalstates)
        lba = (50 * lba) + 5  # scaling magnitude and subtracting offset

        # print("LBA[" + str(i) + "] = " + str(policyaccuracy[i]))

        self.lba[sess_no, trial_no] = lba
        return


    # def test_infogail_two_round_stochastic_env(self):
    #
    #     env = TfEnv(TwoRoundNondeterministicRewardEnv())
    #
    #     # dataset of one-hot obs and acts
    #     # optimal actions: 0, 1
    #     # first state
    #     n_expert_samples = 1000
    #     batch_size = 1000
    #     half = int(n_expert_samples / 2)
    #     rx = np.zeros((n_expert_samples, 3))
    #     rx[:half,2] = 1
    #     rx[half:,0] = 1
    #     ra = np.zeros((n_expert_samples, 2))
    #     ra[:half,0] = 1
    #     ra[half:,1] = 1
    #
    #     with tf.Session() as session:
    #         # critic
    #         critic_dataset = CriticDataset(dict(observations=rx, actions=ra), batch_size=batch_size)
    #         critic_network = ObservationActionMLP(name='critic', hidden_layer_dims=[32,32])
    #         critic = WassersteinCritic(
    #             obs_dim=3,
    #             act_dim=2,
    #             dataset=critic_dataset,
    #             network=critic_network,
    #             gradient_penalty=.01,
    #             optimizer=tf.train.AdamOptimizer(.001, beta1=.5, beta2=.9),
    #             n_train_epochs=50
    #         )
    #
    #         # recognition model
    #         recog_dataset = RecognitionDataset(batch_size=batch_size)
    #         recog_network = ObservationActionMLP(
    #             name='recog',
    #             hidden_layer_dims=[32,32],
    #             output_dim=2
    #         )
    #         recog = RecognitionModel(
    #                     obs_dim=3,
    #                     act_dim=2,
    #                     dataset=recog_dataset,
    #                     network=recog_network,
    #                     variable_type='categorical',
    #                     latent_dim=2
    #         )
    #
    #         # policy
    #         env.spec.num_envs = 10
    #         latent_sampler = UniformlyRandomLatentSampler(
    #             scheduler=ConstantIntervalScheduler(),
    #             name='latent_sampler',
    #             dim=2
    #         )
    #         policy = CategoricalLatentVarMLPPolicy(
    #             policy_name="policy",
    #             latent_sampler=latent_sampler,
    #             env_spec=env.spec
    #         )
    #
    #         # gail
    #         reward_handler = RewardHandler(
    #             use_env_rewards=False,
    #             critic_final_scale=1.
    #         )
    #         baseline = LinearFeatureBaseline(env_spec=env.spec)
    #         algo = GAIL(
    #             critic=critic,
    #             recognition=recog,
    #             reward_handler=reward_handler,
    #             env=env,
    #             policy=policy,
    #             baseline=baseline,
    #             batch_size=4000,
    #             max_path_length=200,
    #             n_itr=15,
    #             discount=.99,
    #             step_size=.01,
    #             sampler_args=dict(n_envs=env.spec.num_envs)
    #         )
    #
    #         session.run(tf.global_variables_initializer())
    #
    #         # run it!
    #         algo.train(sess=session)
    #
    #         # evaluate
    #         l0_state_infos = dict(latent=[[1,0]])
    #         l0_dist_2 = policy.dist_info([[0.,0.,1.]], l0_state_infos)['prob']
    #         l0_dist_0 = policy.dist_info([[1.,0.,0.]], l0_state_infos)['prob']
    #
    #         l1_state_infos = dict(latent=[[0,1]])
    #         l1_dist_2 = policy.dist_info([[0.,0.,1.]], l1_state_infos)['prob']
    #         l1_dist_0 = policy.dist_info([[1.,0.,0.]], l1_state_infos)['prob']
    #
    #         np.testing.assert_array_almost_equal(l0_dist_2, [[1,0]], 1)
    #         np.testing.assert_array_almost_equal(l0_dist_0, [[0,1]], 1)
    #         np.testing.assert_array_almost_equal(l1_dist_2, [[1,0]], 1)
    #         np.testing.assert_array_almost_equal(l1_dist_0, [[0,1]], 1)

if __name__ == '__main__':

    unittest.main()