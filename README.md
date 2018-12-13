# CartPole
Solving CartPole with an Anticipatory Network

## How it works
I came up with this idea after reading how AlphaGo implements a Monte Carlo Tree Search, and thinking about how we approach physical tasks with a mental image of how they should be performed.

In the main loop, the present State is fed into a Multilayer Perceptron (MLP), twice. Once accompanied by the Action for 'move left', and once with 'move right'.

The MLP calculates what it expects the next State to be, and compares it to an ideal (Zen) state of balance – which would be a State of all Zeros (upright, no movement or momentum, and in the middle). It returns the Loss Function of its predicted future State against this 'ideal' State, and chooses the most stable option.

In a sense, it's learning to imagine the task it's trying to perform, while imagining what an ideal performance would look like. No reward necessary.

![alt-text](https://i.imgur.com/UI3nbsg.png)

## Run simulation
As a Markov Model, everything we need to know is in the present State. However, predicting a bit further into the future is now an option. In run_simulation, we can feed the predicted State and current Action back into the MLP, to simulate the network at t+2, t+3, etc.

In this implementation, which has been tuned specifically to solve CartPole-v0 quickly, we look 4 steps ahead. look_ahead_mask decides how much weight to give each prediction, from t to t+4. This kind of tuning is fairly task-specific. However, simply feeding the State through the network four times, and taking the output at t+4, only adds a single episode to our solve time.

Looking ahead seems to have a useful momentum-stabilising effect in this task. In more complex tasks, this could be where a more complete MCTS architecture is implemented, allowing some simulated forward planning.
