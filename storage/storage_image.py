import cv2
from utils import *
import parameters as pms


class Storage(object):
    def __init__(self, agent, env, baseline):
        self.paths = []
        self.env = env
        self.agent = agent
        self.obs = []
        self.obs_origin = []
        self.baseline = baseline
        self.action_history = [[0,0]]*10

    def get_single_path_imp(self):
        self.obs_origin, self.obs, actions, rewards, action_dists = [], [], [], [], []
        # locations = []
        action_historis = []

        action_history = [[0,0]]*10

        ob = self.env.reset()
        ob_r = self.env.render('rgb_array')

        episode_steps = 0
        for _ in xrange(pms.max_path_length):
            self.obs_origin.append(ob)
            deal_ob = self.deal_image(ob)
            action_historis.append(action_history)

            action, action_dist = self.agent.get_action([deal_ob],[action_history])
            action_history = action_history[0:9]
            action_history.append(action)

            self.obs.append(deal_ob)

            actions.append(action)
            action_dists.append(action_dist)
            res = self.env.step(action) # res

            if pms.render:
                self.env.render()
            ob = res[0]
            ob_r = self.env.render('rgb_array')
            rewards.append([res[1]])
            episode_steps += 1
            if res[2]:
                print "break"
                break
        path = dict(
            observations=np.concatenate([self.obs]),
            agent_infos=np.concatenate([action_dists]),
            rewards=np.array(rewards),
            actions=np.array(actions),
            episode_steps=episode_steps,
            action_histories = np.concatenate([action_historis])
            # locations = np.array(locations)
        )
        return path

    def get_single_path(self):
        path = []
        while True:
            path = self.get_single_path_imp()
            if len(path['rewards']) > 3:
                break
            else :
                print "nooooooo generate path too short"
        self.paths.append(path)
        # self.agent.prev_action *= 0.0
        # self.agent.prev_obs *= 0.0
        return path

    def get_paths(self):
        paths = self.paths
        self.paths = []
        return paths

    def process_paths(self, paths):
        # baselines = []
        # returns = []
        sum_episode_steps = 0



        for path in paths:
            sum_episode_steps += path['episode_steps']
            path_baselines = np.append(self.baseline.predict(path), 0)

            deltas = np.concatenate(path["rewards"]) + \
                     pms.discount * path_baselines[1:] - \
                     path_baselines[:-1]


            path["advantages"] = discount(
                deltas, pms.discount * pms.gae_lambda)
            path["returns"] = np.concatenate(discount(path["rewards"], pms.discount))
            # baselines.append(path_baselines[:-1])
            # returns.append(path["returns"])

        # Updating policy.
        action_dist_n = np.concatenate([path["agent_infos"] for path in paths])
        obs_n = np.concatenate([path["observations"] for path in paths])
        action_n = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        # locations = np.concatenate([path["locations"] for path in paths])
        action_histories = np.concatenate([path["action_histories"] for path in paths])

        if pms.center_adv:
            advantages = (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)

        self.baseline.fit(paths)

        samples_data = dict(
            observations=obs_n,
            actions=action_n,
            rewards=rewards,
            advantages=advantages,
            agent_infos=action_dist_n,
            paths=paths,
            sum_episode_steps=sum_episode_steps,
            # locations = locations
            action_histories = action_histories
        )
        return samples_data

    def deal_image(self, image):
        index = len(self.obs_origin)
        image_end = []
        if index<pms.history_number:
            image_end = self.obs_origin[0:index]
            for i in range(pms.history_number-index):
                image_end.append(image)
        else:
            image_end = self.obs_origin[index-pms.history_number:index]

        image_end = np.concatenate(image_end)
        # image_end = image_end.reshape((pms.obs_height, pms.obs_width, pms.history_number))
        obs = cv2.resize(image, (pms.obs_height, pms.obs_width))
        return np.expand_dims(obs, 0)
        # return cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255., (pms.obs_height, pms.obs_width))
