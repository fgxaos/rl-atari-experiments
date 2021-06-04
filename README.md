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

- Install ROMs

In order to import ROMS, you need to download `Roms.rar` from the [Atari 2600 VCS ROM Collection](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) and extract the `.rar` file.  Once you've done that, run:

`python -m atari_py.import_roms <path to folder>`

This should print out the names of ROMs as it imports them.  The ROMs will be copied to your `atari_py` installation directory.

- Run the experiment
```
python main.py
```

_Note: in our experiments we used only three Atari environments (`Freeway-v0`, `Skiing-v0`, `MsPacman-v0`); but it is possible to run an experiment with any other Atari game (as long as it is available in the gym environment)._

## References
In this project, we used:
- some functions from [OpenAI's baselines](https://github.com/openai/baselines);
- code from PyTorch's tutorial on the [DQN model](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html);
- code from Facebook Research's [MNF-DQN model](https://github.com/facebookresearch/RandomizedValueFunctions).
