# RL Atari experiments
> Deep Reinforcement Learning project with Atari games for the Reinforcement Learning course at CentraleSupelec

## Description
For this project, we implemented two different models: [DQN](https://arxiv.org/abs/1312.5602) and [MNF-DQN](https://arxiv.org/abs/1806.02315).

Thanks to OpenAI's [`gym`](https://github.com/openai/gym) environment, any [Atari environment](https://gym.openai.com/envs/#atari) can be used to train one of the two aforementioned models.

## How to run
- Install the required libraries
```
pip install -r requirements.txt
```
- Set the configuration file `cfg.yml` to run the desired experiment
- Run the experiment
```
python main.py
```

## References
In this project, we used:
- some functions from [OpenAI's baselines](https://github.com/openai/baselines);
- code from PyTorch's tutorial on the [DQN model](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html);
- code from Facebook Research's [MNF-DQN model](https://github.com/facebookresearch/RandomizedValueFunctions).