from util import Counter
from random import Random


class QAgent:

  def __init__(self, transition_func, reward_func, init_state, all_actions):
    self.Q = Counter()

    self.all_actions = all_actions

    # state and action in last round
    self.state = init_state
    self.action = self.all_actions[0]

    # functions
    self.get_reward = reward_func
    self.get_next_state = transition_func

    # q-learning parameters
    self.gamma = 0.1
    self.epsilon = 0.1

    self.rand = Random()

  def observe_and_act(self, observation):
    reward = self.get_reward(self.state, self.action, observation)
    next_state = self.get_next_state(self.state, self.action, observation)
    self._update_q(self.state, self.action, next_state, reward)
    self.state = next_state

    if self.rand.uniform(0, 1) < self.epsilon:
      action_id = int(self.rand.uniform(0, len(self.all_actions)))
      self.action = self.all_actions[action_id]
    else:
      _, self.action = self._get_max_q_value(self.state)
    return self.action, reward

  def _get_max_q_value(self, state):
    tmp_max, tmp_action = 0, self.all_actions[0]
    for action in self.all_actions:
      if self.Q[(state, action)] > tmp_max:
        tmp_max = self.Q[(state, action)]
        tmp_action = action
    return tmp_max, tmp_action

  def _update_q(self, state, action, next_state, reward):
    max_q, _ = self._get_max_q_value(next_state)
    self.Q[(state, action)] = reward + self.gamma * max_q
