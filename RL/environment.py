import random

from config import _DEBUG_NO_NOISE


class _Noise:

  def __init__(self, n_channels, n_states=4):
    self.n_states = int(n_states)
    self.n_channels = n_channels
    self.current_state = 0

    self.state_noise = [[] for i in xrange(self.n_states)]
    self.random = random.Random()
    for i in xrange(self.n_states):
      self.state_noise[i] = [self.random.random()
                             for j in xrange(self.n_channels + 1)]
      amount = 0
      for j in xrange(self.n_channels + 1):
        amount += self.state_noise[i][j]
      for j in xrange(self.n_channels + 1):
        if _DEBUG_NO_NOISE:
          self.state_noise[i][j] = 0
        else:
          self.state_noise[i][j] /= float(amount) #normlize

  def get_state(self):
    self.current_state = (self.current_state + 1) % self.n_states
    return self.state_noise[self.current_state]

#purpose?
def _get_distance_square(pos1, pos2=(0, 0, 0)):
  assert len(pos1) == 3 and len(pos2) == 3
  return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + \
      (pos1[2] - pos2[2]) ** 2


class Environment:

  def __init__(self, n_nodes, n_channels, n_noises=-1, min_distance=1,
               max_distance=10, const_parameter=True):
    self.n_nodes = n_nodes
    self.n_channels = n_channels
    self.n_noises = int(
        self.n_channels * (2 if n_noises == -1 else n_noises)) #why?

    self.random = random.Random()
    self.next_int = lambda: self.random.uniform(min_distance,
                                                max_distance)
    self.next_float = self.random.random

    self.noises = [_Noise(self.n_channels, self.next_int())
                   for i in xrange(self.n_noises)]

    if const_parameter:
      self.node_pos = [(1, 1, 1) for i in xrange(self.n_nodes)]
      self.noise_pos = [(1, 1, 1) for i in xrange(self.n_noises)]
      self.node_power = [3 for i in xrange(self.n_nodes)]
      self.noise_power = [1 for i in xrange(self.n_noises)]
    else:
      self.node_pos = [(0.5 + self.next_float(), 0.5 + self.next_float(),
                        0.5 + self.next_float())
                       for i in xrange(self.n_nodes)]
      self.noise_pos = [(1 + self.next_float(), 1 + self.next_float(),
                         1 + self.next_float())
                        for i in xrange(self.n_noises)]
      self.node_power = [self.next_float() for i in xrange(self.n_nodes)]
      self.noise_power = [self.next_float() for i in xrange(self.n_noises)]

  def _get_noise_channel_probability(self):
    return [self.noises[i].get_state()
            for i in xrange(self.n_noises)]

  def _get_channel_noise_mapping(self):
    self.noise_channel_probability = self._get_noise_channel_probability()
    # generate noise affect table
    noise_channel_mapping = [-1 for j in xrange(self.n_noises)]
    for j in xrange(self.n_noises):
      tmp = 1 if _DEBUG_NO_NOISE else self.next_float()
      present = 0
      for i in xrange(self.n_channels):
        present += self.noise_channel_probability[j][i]
        if tmp < present:
          noise_channel_mapping[j] = i
          break
      # if noise_channel_mapping[j] == -1:
      #  noise_channel_mapping[j] = self.n_channels - 1

    channel_noise = [[] for j in xrange(self.n_channels)]
    for j in xrange(self.n_noises):
      if noise_channel_mapping[j] != -1:
        channel_noise[noise_channel_mapping[j]].append(j)

    return channel_noise

  def _get_conflict_matrix(self):
    conflict_matrix = [[0 for j in xrange(self.n_nodes)]
                       for i in xrange(self.n_channels)]

    channel_noise = self._get_channel_noise_mapping()
    for i in xrange(self.n_channels):
      noise_list = channel_noise[i]
      noise_power = 0
      for j in noise_list:
        assert type(j) == int
        noise_power += self.noise_power[j] / \
            float(_get_distance_square(self.noise_pos[j]))

      for j in xrange(self.n_nodes):
        node_power = self.node_power[j] / \
            float(_get_distance_square(self.node_pos[j]))
        if node_power > noise_power:
          conflict_matrix[i][j] = 0
        else:
          conflict_matrix[i][j] = 1

    return conflict_matrix

  def process(self, action):
    assert len(action) == self.n_nodes and type(action[0]) == int

    conflict_matrix = self._get_conflict_matrix()
    observation = [0 for i in xrange(self.n_nodes)]
    for node_id in xrange(self.n_nodes):
      channel_id = action[node_id]
      if channel_id != -1 and conflict_matrix[channel_id][node_id] == 0:
        observation[node_id] = 1

    result = tuple(observation)
    return result




