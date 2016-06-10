# basic settings
"""

Denote s = N_NODES, c = N_CHANNELS, w = AGENT_STATE_WINDOWS_SIZE.

complexity:
  #(observation): pow(2, s)
  #(state): pow(2, ws)
  #(action): pow(c + 1, s)
"""
N_NODES = 4
N_NOISES = 0.5
N_CHANNELS = 1

# q-learning settings
AGENT_STATE_WINDOWS_SIZE = 5

# test settings
STAGES = 100
ROUNDS = 100

# debug settings
OUTPUT_F = 'result.txt'
_DEBUG_NO_NOISE = True
