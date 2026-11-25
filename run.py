"""Hydra entry point for training, including experiment=pc/am_pc_edge."""

from rl4co.tasks.train import train

if __name__ == "__main__":
    train()
