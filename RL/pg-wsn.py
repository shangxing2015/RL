""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from pg_environment import Environment
import math

#===================================
N_NODES = 1
N_NOISES = 0.5
N_CHANNELS = 1


#====================================
# q-learning settings
AGENT_STATE_WINDOWS_SIZE = 40

# hyperparameters
H = 5# number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?


# model initialization
D = N_CHANNELS* AGENT_STATE_WINDOWS_SIZE # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    #if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  print discounted_r
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}


#================create environment and obseravtion======================


env = Environment(N_NODES, N_CHANNELS, N_NOISES, min_distance=1,
               max_distance=10, const_parameter=True)

init_state = np.zeros((AGENT_STATE_WINDOWS_SIZE,N_CHANNELS))
state = init_state


prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
count = batch_size
done = False

iter_num = 5000
rec = 0
observation = np.zeros(N_CHANNELS).reshape((1,N_CHANNELS))

rec_prob = list()
rec_reward = list()

while iter_num > 0:

  rec += 1
  count = count - 1

  state = np.concatenate((state[1:][0:],observation),axis = 0)
  x = np.ravel(state)


  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  rec_prob.append(aprob)
  #print 'prob: ' + str(aprob)

  action = 1 if np.random.uniform() < aprob else 0 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 1 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  temp = list()
  temp.append(action)
  #print 'action: ' + str(action)
  observation, reward = env.step(temp, rec)
  reward_sum += reward

  rec_reward.append(reward_sum/float(rec))

  drs.append(float(reward if reward >0 else -1)) # record reward (has to be done after we call step() to get reward for previous action)


  #print str(rec) + ': ' + str(reward_sum/float(rec))

  if count == 0:
      count = batch_size
      done = True

  if done: # an episode finished
    #print 'enter'
    iter_num -= 1
    done = False
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    #print 'epx: ' + str(epx)
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp)
    for k in model:
        grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes

    for k,v in model.iteritems():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    #print 'end'
    #print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)


y=range(1,len(rec_prob)+1)
print 'average prob: ' + str(np.mean(rec_prob))
print 'final prob: ' + str(np.mean(rec_prob[len(y)-50:]))
print 'average prob: ' + str(np.mean(rec_reward))
print 'final prob: ' + str(np.mean(rec_reward[len(y)-50:]))
plt.figure(1)
plt.subplot(211)
plt.plot(y,rec_prob)
plt.ylabel('prob of using channel')
plt.axis([1, len(rec_prob), 0, 1])


plt.subplot(212)
plt.plot(y,rec_reward)
plt.ylabel('time average reward')
plt.show()





