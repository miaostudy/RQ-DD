#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from diffusers.models import AutoencoderKL
from omegaconf import DictConfig
from PIL import Image

from qinco.qinco_tasks import QincoConvertTask, QincoEvalTask, QincoTrainTask
from qinco.search.search_tasks import (
    BuildIndexTask,
    EncodeDBTask,
    IVFTrainTask,
    SearchTask,
    TrainPairwiseDecoderTask,
)

import os
os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"

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


def _infer_latent_hw_from_d(d: int, latent_channels: int) -> tuple[int, int] | None:
    if d % latent_channels != 0:
        return None
    spatial = d // latent_channels
    side = int(spatial**0.5)
    if side * side != spatial:
        return None
    return side, side


def _save_grid_png(images_kchw: torch.Tensor, out_path: Path, nrow: int) -> None:
    # images_kchw assumed in [0, 1]
    k, _c, h, w = images_kchw.shape
    ncol = int(np.ceil(k / nrow))
    canvas = np.zeros((ncol * h, nrow * w, 3), dtype=np.uint8)
    imgs = (images_kchw.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    imgs = np.transpose(imgs, (0, 2, 3, 1))  # K, H, W, C
    for idx in range(k):
        r = idx // nrow
        c = idx % nrow
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = imgs[idx]
    Image.fromarray(canvas).save(out_path)


def _load_vae_for_export(cfg: DictConfig, device: torch.device) -> AutoencoderKL:
    model_id = str(cfg.vae.model_id)
    try:
        vae = AutoencoderKL.from_pretrained(model_id).to(device)
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "Failed to load VAE for codebook visualization. "
            "Ensure network/proxy is available (for example 127.0.0.1:7890) and retry. "
            f"model_id={model_id}"
        ) from exc
    vae.eval()
    return vae


def export_main_codebooks(output_pt: str, out_dir: Path, cfg: DictConfig) -> None:
    state_dict = torch.load(output_pt, map_location="cpu", weights_only=True)
    model_state = state_dict["model"]
    latent_channels = int(cfg.vae.latent_channels)
    scale_factor = float(cfg.vae.scale_factor)
    input_image_size = int(cfg.vae.input_image_size)
    expected_latent_side = input_image_size // 8
    expected_d = latent_channels * expected_latent_side * expected_latent_side
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = _load_vae_for_export(cfg, device)

    codebooks = []
    i_step = 0
    while True:
        key = f"steps.{i_step}.codebook.weight"
        if key not in model_state:
            break
        codebooks.append(model_state[key].detach().cpu())
        i_step += 1

    if not codebooks:
        raise RuntimeError("No main codebooks found in checkpoint.")

    save_payload = {
        "n_steps": len(codebooks),
        "shapes": [tuple(cb.shape) for cb in codebooks],
        "codebooks": codebooks,
    }
    torch.save(save_payload, out_dir / "codebooks_main.pt")

    for step_idx, cb in enumerate(codebooks):
        k, d = cb.shape
        if d == expected_d:
            h = expected_latent_side
            w = expected_latent_side
        else:
            inferred_hw = _infer_latent_hw_from_d(d, latent_channels)
            if inferred_hw is None:
                print(
                    f"Skip visualization for step {step_idx}: D={d} is not "
                    f"{latent_channels}*H*W square latent format."
                )
                continue
            h, w = inferred_hw
            print(
                f"Warning: step {step_idx} has D={d}, expected D={expected_d} for "
                f"input_image_size={input_image_size}. Using inferred latent size {h}x{w}."
            )
        latents = cb.reshape(k, latent_channels, h, w).to(device)
        decoded_batches = []
        with torch.no_grad():
            for chunk in torch.split(latents, 16, dim=0):
                decoded = vae.decode(chunk / scale_factor).sample
                decoded_batches.append(decoded.cpu())
        imgs = torch.cat(decoded_batches, dim=0)
        imgs = ((imgs + 1.0) / 2.0).clamp(0.0, 1.0)

        nrow = max(1, int(k**0.5))
        _save_grid_png(imgs, out_dir / f"codebook_step{step_idx}.png", nrow=nrow)


def resolve_class_output_dir(cfg: DictConfig) -> Path:
    base_dir = Path(cfg.output).resolve().parent
    trainset_path = Path(str(cfg.trainset)).resolve() if cfg.trainset is not None else None
    if trainset_path is not None and trainset_path.name == "vectors.npy":
        class_name = trainset_path.parent.name
    elif trainset_path is not None:
        class_name = trainset_path.stem
    else:
        class_name = "all_classes"
    return base_dir / class_name


@hydra.main(version_base=None, config_path="config", config_name="qinco_cfg")
def main(cfg: DictConfig):
    print(cfg)
    if cfg.task is None:
        raise ValueError(
            "Please specify a task (train, eval, etc.) using the 'train=<...>' argument"
        )
    expe = EXPERIMENTS[cfg.task](cfg)

    expe.accelerator.print(f"====================== RUNNING TASK {cfg.task}")
    expe.run()

    if cfg.task == "train":
        expe.accelerator.wait_for_everyone()
        if expe.accelerator.is_main_process:
            out_dir = resolve_class_output_dir(cfg)
            out_dir.mkdir(parents=True, exist_ok=True)
            export_main_codebooks(cfg.output, out_dir, cfg)
            expe.accelerator.print(f"Exported main codebooks to {out_dir}")

    expe.accelerator.print("Task done")
    expe.accelerator.end_training()


if __name__ == "__main__":
    main()  # pylint: disable=all
