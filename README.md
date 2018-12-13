# CartPole
Solving CartPole by simulating CartPole in a deep network

## How it works
I came up with this idea after reading about how AlphaGo implements a Monte Carlo Tree Search, and thinking about how we approach certain tasks utilising a mental image of how they should be performed.

In the main loop, the present State is fed into a Multilayer Perceptron (MLP), twice. Once accompanied by the Action for 'move left', and once with 'move right'.

The MLP calculates what it expects the next State to be, and compares it to an ideal (Zen) state of balance – which would be a State of all Zeros (upright, no movement or momentum, and in the middle). It returns the Loss Function of its predicted future State against this 'ideal' State, and chooses the most stable option.

In a sense, it's effectively learning to imagine the task it's trying to perform, while imagining what an ideal performance would look like.

![alt-text](https://i.imgur.com/UI3nbsg.png)

## Run simulation
As a Markov Model, everything we need to know is in the present State. However, predicting a bit further into the future is now an option. In run_simulation, we can feed the predicted State and current Action back into the MLP, to simulate the network at t+2, t+3, etc.

In this implementation, which has been tuned specifically to solve CartPole-v0 quickly, we look up to 4 steps ahead. look_ahead_mask decides how much weight to give each prediction, from t to t+4. This kind of tuning is fairly task-specific, and a bit of a 'cheat'. However, simply feeding the State through the network four times, and taking the output at t+4, only adds a single episode to our solve time.

This looking ahead seems to have a momentum-stabilising effect. Obviously this is where a more complex task would benefit from some conventional MCTS architecture. And this is where I expect the real potential of this approach to be.
