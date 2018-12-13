# CartPole
Solving CartPole by simulating CartPole in a deep network

## How it works
I came up with this idea after reading about how AlphaGo uses a Monte Carlo Tree Search.

In the case of CartPole, with analog States, a conventional MCTS is hard to implement. We could quantise the State data, but then learning becomes fragmented – learning how to balance in the middle of the path becomes a separate task from learning how to balance at the sides.

So in this case, a neural net would have to power our Decision Tree. And therefore what this network would be doing is predicting what State comes next.

In the main loop, the present State is fed into a Multilayer Perceptron (MLP), twice. Once with the Action for 'move left', and once with 'move right'.

The MLP calculates what it expects the next State to be, and compares it to an ideal (Zen) state of balance – which would be a State of all Zeros (upright, no movement or momentum, and in the middle). It simply returns the Loss Function of its predicted future State against this 'ideal' State.

In a sense, it's learning to imagine the task it's trying to perform, while imagining what an ideal performance would look like.

## Run simulation
As a Markov Model, everything we need to know is in the present State. However, predicting a bit further into the future is now an option. In run_simulation, we can feed the predicted State back into the MLP, to simulate the network at t+2, t+3, etc.

In this implementation, which has been tuned to solve CartPole-v0 quickly, we look up to 4 steps ahead. look_ahead_mask decides how much weight to give each prediction, from t to t+4. This kind of tuning is fairly task-specific, and a bit of a 'cheat'. However, simply feeding the State through the network four times and taking the output at t+4 only adds a single episode to our solve time.

Looking ahead seems to have a momentum-stabilising effect. Obviously this is where a more complex task would benefit from a conventional MCTS architecture. And this is where I expect the real potential of this approach to be.

