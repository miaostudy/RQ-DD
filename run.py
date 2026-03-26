# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from omegaconf import DictConfig

# Imports experiments (necessary to register experiments)
from qinco.qinco_tasks import QincoConvertTask, QincoEvalTask, QincoTrainTask
from qinco.search.search_tasks import (
    BuildIndexTask,
    EncodeDBTask,
    IVFTrainTask,
    SearchTask,
    TrainPairwiseDecoderTask,
)

EXPERIMENTS = {
    "train": QincoTrainTask,
    "eval_valset": QincoTrainTask,
    "eval": QincoEvalTask,
    "eval_time": QincoEvalTask,
    "convert": QincoConvertTask,
    "ivf_centroids": IVFTrainTask,
    "encode": EncodeDBTask,
    "build_index": BuildIndexTask,
    "train_pairwise_decoder": TrainPairwiseDecoderTask,
    "search": SearchTask,
}

# hydra的包装器，会在运行main之前自动解析配置和命令行中的参数，并以cfg: DictConfig传入main
@hydra.main(version_base=None, config_path="config", config_name="qinco_cfg")
def main(cfg: DictConfig):
    print(cfg)
    if cfg.task is None:
        raise ValueError(
            "Please specify a task (train, eval, etc.) using the 'train=<...>' argument"
        )
    expe = EXPERIMENTS[cfg.task](cfg) # 任务初始化

    expe.accelerator.print(f"====================== RUNNING TASK {cfg.task}")
    expe.run()
    expe.accelerator.print("Task done")
    expe.accelerator.end_training()  # Destroy process group


if __name__ == "__main__":
    main()  # pylint: disable=all
