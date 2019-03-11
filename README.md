# CartPole
Solving CartPole with an Anticipatory Network

## How it works
I came up with this idea after reading how AlphaGo implements a Monte Carlo Tree Search, and pondering how we approach tasks with a mental image of how they should be performed.

In the main loop, the present State is fed into a Multilayer Perceptron (MLP) twice. Once with the Action for 'move left', and once with 'move right'.

The MLP calculates what it expects the next State to be, based on this input, and compares it to an ideal state of all Zeros (upright, no movement or momentum, and in the middle). It returns the Loss Function of its predicted future State against the 'ideal' State, and chooses the Action path that gets it closest.

In a sense, it's learning to imagine the task it's trying to perform, while picturing what an ideal performance would look like. No reward necessary. The only thing that defines its success is how well it can estimate the next State with each hypothetical action.

![alt-text](https://i.imgur.com/UI3nbsg.png)

## Run simulation
As a Markov Model, everything we need to know is in the present State. However, as we're predicting one step ahead, predicting a bit further into the future is now an option. In run_simulation, we can feed the predicted State and current Action back into the MLP, to simulate the network at t+2, t+3, etc.

In this implementation, which has been tuned specifically to solve CartPole-v0, we look 4 steps ahead. A 'look ahead' mask decides how much weight to give each time step (t+1 to t+4). This tuning could be fairly task-specific. However, simply feeding the State through the network four times, and taking the output at t+4, only adds a single episode to our solve time.

Looking ahead seems to have a useful momentum-stabilising effect in this task. In more complex tasks, this could house a more complete MCTS architecture.
