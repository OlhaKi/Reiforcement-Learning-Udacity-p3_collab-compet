# Project 3 : Collaboration and Competition
The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.
### Project Overview
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.<br> 
Thus, the goal of each agent is to keep the ball in play.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

  * After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
  * This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.<br>

### Learning algorithm
For this project I have used a  Multi Agent Deep Deterministic Policy Gradient (MADDPG) which is described in the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.](https://arxiv.org/abs/1706.02275)<br>
Multi-Agent Deep Determinisitc Policy Gradient(MADDPG)<br>

Pseudocode:<br>
![Pseudo](/Images/maddpg-algo.png)

The main concept this algorithm is summarized in this illustration:
![Main concept](/Images/Multi-Agent-DDPG-Actor.png)
<br>The Critics networks have access to the states and actions information of both agents, while the Actors networks have only access to the information corresponding to their local agent.<br>
A plot of rewards per episode is included to illustrate that the agents get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).
The submission reports the number of episodes needed to solve the environment.
The submission has concrete future ideas for improving the agent's performance.

#### Concrete future ideas:
Create Munchausen agents(see [Munchausen Reinforcement Learning](https://arxiv.org/abs/2007.14430)).<br>
Try dropout / different optimizer to stabilize learning.<br>
Try out other Actor-Critic method variants (for example, A3C).<br>
Try other noise process algorithms.<br>
