from copy import deepcopy
from pathlib import Path

import hydra
import lightning as L
import pyrootutils
import torch

from lightning import Callback, LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from rl4co import utils
from rl4co.utils import RL4COTrainer

pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)


log = utils.get_pylogger(__name__)


@utils.task_wrapper
def run(cfg: DictConfig) -> tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.
    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # We instantiate the environment separately and then pass it to the model
    log.info(f"Instantiating environment <{cfg.env._target_}>")
    env = hydra.utils.instantiate(cfg.env)

    # Note that the RL environment is instantiated inside the model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, env)

    init_checkpoint = cfg.get("init_from_checkpoint")
    if init_checkpoint:
        log.info(f"Initializing model weights from <{init_checkpoint}>")
        checkpoint = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing_policy = [key for key in incompatible.missing_keys if key.startswith("policy.")]
        unexpected_policy = [
            key for key in incompatible.unexpected_keys if key.startswith("policy.")
        ]
        if missing_policy or unexpected_policy:
            raise RuntimeError(
                "Policy checkpoint is incompatible with the curriculum stage. "
                f"Missing policy keys: {missing_policy}; "
                f"unexpected policy keys: {unexpected_policy}"
            )
        log.info(
            "Loaded weights only; optimizer state and epoch counter will start fresh. "
            f"Non-policy missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = utils.instantiate_loggers(cfg.get("logger"), model)

    log.info("Instantiating trainer...")
    trainer: RL4COTrainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile", False):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        # Full-state curriculum checkpoints contain OmegaConf objects and
        # optimizer/callback state.  PyTorch 2.6 changed torch.load's default
        # to weights_only=True, which rejects those trusted local checkpoints.
        # Explicitly request a full restore when resuming training.
        trainer.fit(
            model=model,
            ckpt_path=cfg.get("ckpt_path"),
            weights_only=False if cfg.get("ckpt_path") else None,
        )

        train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, ckpt_path=ckpt_path, weights_only=False)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


def train_pc_curriculum(cfg: DictConfig) -> None:
    """Train three PC stages per resolution, warm-starting from the best policy."""
    if not cfg.train or cfg.ckpt_path or cfg.init_from_checkpoint:
        raise ValueError(
            "The PC curriculum starts from scratch. Set curriculum.enabled=false "
            "for an individual training stage, resume, or test-only run."
        )

    bands = [[10, 20], [21, 30], [31, 40]]
    stages = [
        ("stage1_n10-20", [1.0]),
        ("stage2_n10-30", [0.2, 0.8]),
        ("stage3_n10-40", [0.1, 0.1, 0.8]),
    ]
    root = Path(cfg.paths.output_dir)
    for gamma in cfg.curriculum.gammas:
        gamma_name = f"{gamma:g}".replace(".", "p")
        previous_best = None
        for stage_number, (stage_name, weights) in enumerate(stages, start=1):
            stage_dir = root / f"gamma_{gamma_name}" / stage_name
            stage_dir.mkdir(parents=True, exist_ok=False)
            stage = deepcopy(cfg)
            stage.curriculum.enabled = False
            stage.paths.output_dir = str(stage_dir)
            stage.env.modularity_gamma = gamma
            num_parts = bands[stage_number - 1][1]
            generator = stage.env.generator_params
            generator.num_parts = bands[0][0]
            generator.max_num_parts = num_parts
            generator.size_ranges = bands[:stage_number]
            generator.size_weights = weights
            stage.env.val_file = f"pc{num_parts}_newdist_val_seed4321.npz"
            stage.env.test_file = f"pc{num_parts}_newdist_test_seed1234.npz"
            if stage_number > 1 and gamma >= 0.5:
                stage.model.optimizer_kwargs.lr = cfg.curriculum.later_stage_lr
            stage.logger.tensorboard.name = f"curriculum_gamma{gamma_name}_stage{stage_number}"
            stage.init_from_checkpoint = previous_best

            log.info(f"Training PC curriculum: gamma={gamma:g}, stage={stage_number}")
            _, objects = run(stage)
            callback = objects["trainer"].checkpoint_callback
            previous_best = callback.best_model_path if callback else ""
            if not previous_best or not Path(previous_best).is_file():
                raise RuntimeError(f"No best policy checkpoint produced in {stage_dir}")
            del objects, callback


@hydra.main(version_base="1.3", config_path="../../configs", config_name="main.yaml")
def train(cfg: DictConfig) -> float | None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    if cfg.get("curriculum") and cfg.curriculum.enabled:
        train_pc_curriculum(cfg)
        return None

    # train the model
    metric_dict, _ = run(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    train()
