### LIBRARIES ###
# Global libraries
import yaml
import torch
import wandb

# Custom libraries
from models.dqn.dqn_agent import DQNAgent
from models.mnfdqn.mnfdqn_agent import MNFDQNAgent

### UTILS FUNCTION ###
def run_experiment(cfg):
    wandb.init(project="rl-atari-experiments", config=cfg)

    if cfg["model"] == "dqn":
        DQNAgent(
            env=cfg["env"],
            n_episodes=cfg["n_episodes"],
            device=cfg["device"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            gamma=cfg["gamma"],
            eps_start=cfg["eps_start"],
            eps_end=cfg["eps_end"],
            eps_decay=cfg["eps_decay"],
            target_update=cfg["target_update"],
            train_render=cfg["train_render"],
            test_render=cfg["test_render"],
            initial_memory=cfg["initial_memory"],
        )

    elif cfg["model"] == "mnfdqn":
        agent = MNFDQNAgent(
            env=cfg["env"],
            n_episodes=cfg["n_episodes"],
            device=cfg["device"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            discount=cfg["discount"],
            double_q=cfg["double_q"],
            alpha=cfg["alpha"],
            replay_buffer_size=cfg["replay_buffer_size"],
            hidden_dim=cfg["hidden_dim"],
            n_hidden=cfg["n_hidden"],
            n_flows_q=cfg["n_flows_q"],
            n_flows_r=cfg["n_flows_r"],
            model=cfg["model"],
            target_update_freq=cfg["target_update_freq"],
            learning_starts=cfg["learning_starts"],
            learning_freq=cfg["learning_freq"],
            render=cfg["render"],
        )
        agent.train()

    else:
        raise ValueError(f"No model with the name {cfg['model']}")


### MAIN CODE ###
with open("cfg.yml", "r") as yml_file:
    yml_data = yaml.safe_load(yml_file)

cfg = yml_data[yml_data["model"]]
cfg["env"] = yml_data["env"]
cfg["model"] = yml_data["model"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg["device"] = device

run_experiment(cfg)
