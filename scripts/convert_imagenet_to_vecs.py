#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from numpy.lib.format import open_memmap

import os
os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
SPEC_TO_CLASS_FILE = {
    "nette": "class_nette.txt",
    "woof": "class_woof.txt",
    "1k": "class_indices.txt",
    "100": "class100.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ImageNet images to flattened VAE latent vectors (N, D) stored as .npy."
    )
    parser.add_argument("--imagenet-root", type=Path, default=Path("/data/wlf/datasets/imagenet"), help="ImageNet root directory.")
    parser.add_argument(
        "--misc-dir",
        type=Path,
        default=Path("../IGD/misc"),
        help="Directory containing class_*.txt files.",
    )
    parser.add_argument(
        "--spec",
        type=str,
        default="nette",
        choices=["nette", "woof", "1k", "100"],
        help="Class subset spec.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "all"],
        help="Which split(s) to convert.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Target image size for Resize + CenterCrop. Must be divisible by 8 for SD-VAE latent shape.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/imagenet_vectors"),
        help="Output directory. Result will be organized as <out-dir>/<class-name>/...",
    )
    parser.add_argument(
        "--out-npy",
        type=Path,
        default=None,
        help="Optional single merged output .npy path for all classes.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If > 0, only convert first N matched images.",
    )
    parser.add_argument(
        "--save-labels",
        action="store_true",
        help="Also save integer class labels beside output as <out>_labels.npy.",
    )
    parser.add_argument(
        "--save-per-sample",
        action="store_true",
        help="Also save one .npy per image inside each class directory.",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default="mse",
        help="VAE preset used by IGD. Used as stabilityai/sd-vae-ft-{vae} when --vae-model is not set.",
    )
    parser.add_argument(
        "--vae-model",
        type=str,
        default=None,
        help="Optional explicit HF model id for AutoencoderKL.",
    )
    parser.add_argument(
        "--vae-batch-size",
        type=int,
        default=64,
        help="Batch size used for VAE encoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for VAE encoding.",
    )
    return parser.parse_args()


def read_class_list(misc_dir: Path, spec: str) -> list[str]:
    class_file = misc_dir / SPEC_TO_CLASS_FILE[spec]
    if not class_file.exists():
        raise FileNotFoundError(f"Class file not found: {class_file}")
    classes = [line.strip() for line in class_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"No classes found in {class_file}")
    return classes


def iter_split_roots(imagenet_root: Path, split: str) -> Iterable[Path]:
    if split in {"train", "val"}:
        roots = [imagenet_root / split]
    else:
        roots = [imagenet_root / "train", imagenet_root / "val"]
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Split directory not found: {root}")
        yield root


def collect_image_paths(split_roots: Iterable[Path], class_names: list[str]) -> tuple[list[Path], list[int]]:
    file_paths: list[Path] = []
    labels: list[int] = []
    for split_root in split_roots:
        for class_idx, class_name in enumerate(class_names):
            class_dir = split_root / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
                    file_paths.append(path)
                    labels.append(class_idx)
    if not file_paths:
        raise ValueError("No images were found for the requested split/spec.")
    return file_paths, labels


def collect_image_paths_by_class(
    split_roots: Iterable[Path], class_names: list[str]
) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = {class_name: [] for class_name in class_names}
    for split_root in split_roots:
        for class_name in class_names:
            class_dir = split_root / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS:
                    by_class[class_name].append(path)
    return by_class


def resize_center_crop_rgb(img: Image.Image, image_size: int) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    scale = image_size / min(w, h)
    new_w = max(image_size, int(round(w * scale)))
    new_h = max(image_size, int(round(h * scale)))
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    left = (new_w - image_size) // 2
    top = (new_h - image_size) // 2
    img = img.crop((left, top, left + image_size, top + image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, C) in [0, 1]
    arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
    return arr


def resolve_vae_model_id(args: argparse.Namespace) -> str:
    if args.vae_model:
        return args.vae_model
    return f"stabilityai/sd-vae-ft-{args.vae}"


def load_vae(model_id: str, device: torch.device) -> AutoencoderKL:
    try:
        vae = AutoencoderKL.from_pretrained(model_id).to(device)
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "Failed to load VAE model. If this is the first run, ensure network/proxy is available "
            "(for example, set http_proxy/https_proxy to 127.0.0.1:7890), "
            f"then retry. model_id={model_id}"
        ) from exc
    vae.eval()
    return vae


def convert_class_to_dir(
    class_name: str,
    image_paths: list[Path],
    out_dir: Path,
    image_size: int,
    save_per_sample: bool,
    vae: AutoencoderKL,
    vae_batch_size: int,
    device: torch.device,
) -> tuple[int, float, float]:
    class_dir = out_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    n_samples = len(image_paths)
    if image_size % 8 != 0:
        raise ValueError(f"image_size must be divisible by 8 for SD-VAE. Got {image_size}")
    latent_h = image_size // 8
    latent_w = image_size // 8
    d = 4 * latent_h * latent_w
    class_vec_path = class_dir / "vectors.npy"
    class_vecs = open_memmap(
        str(class_vec_path), mode="w+", dtype=np.float32, shape=(n_samples, d)
    )

    running_sum = 0.0
    running_sq_sum = 0.0
    total_vals = float(max(n_samples * d, 1))

    rel_paths = []
    idx = 0
    with torch.no_grad():
        while idx < n_samples:
            batch_paths = image_paths[idx : idx + vae_batch_size]
            batch_chw = []
            for img_path in batch_paths:
                with Image.open(img_path) as img:
                    chw = resize_center_crop_rgb(img, image_size)
                # SD-VAE expects input in [-1, 1].
                chw = chw * 2.0 - 1.0
                batch_chw.append(chw)
            x = torch.from_numpy(np.stack(batch_chw, axis=0)).to(device=device, dtype=torch.float32)
            posterior = vae.encode(x).latent_dist
            latents = posterior.mode() * 0.18215
            latents_np = latents.detach().cpu().numpy().astype(np.float32, copy=False)
            flat_batch = latents_np.reshape(latents_np.shape[0], -1)

            for j in range(flat_batch.shape[0]):
                sample_idx = idx + j
                flat = flat_batch[j]
                class_vecs[sample_idx] = flat
                if save_per_sample:
                    np.save(class_dir / f"{sample_idx:06d}.npy", flat)
                rel_paths.append(str(batch_paths[j]))
                running_sum += float(flat.sum())
                running_sq_sum += float((flat * flat).sum())

            idx += flat_batch.shape[0]
            if idx % 2000 == 0 or idx == n_samples:
                print(f"[{class_name}] [{idx}/{n_samples}] converted")

    np.save(class_dir / "source_paths.npy", np.asarray(rel_paths, dtype=object))

    mean = running_sum / total_vals
    var = max(running_sq_sum / total_vals - mean * mean, 0.0)
    std = var**0.5
    return n_samples, mean, std


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model_id = resolve_vae_model_id(args)
    print(f"Loading VAE: {model_id} on {device}")
    vae = load_vae(model_id, device)

    class_names = read_class_list(args.misc_dir, args.spec)
    split_roots = list(iter_split_roots(args.imagenet_root, args.split))
    by_class = collect_image_paths_by_class(split_roots, class_names)

    if args.max_samples > 0:
        for class_name in class_names:
            by_class[class_name] = by_class[class_name][: args.max_samples]

    class_summary = []
    for class_name in class_names:
        image_paths = by_class[class_name]
        if not image_paths:
            print(f"[{class_name}] no images found, skipped")
            continue
        n_samples, mean, std = convert_class_to_dir(
            class_name,
            image_paths,
            args.out_dir,
            args.image_size,
            args.save_per_sample,
            vae,
            args.vae_batch_size,
            device,
        )
        class_summary.append((class_name, n_samples, mean, std))

    if not class_summary:
        raise ValueError("No class has images to convert.")

    if args.out_npy is not None:
        file_paths, labels = collect_image_paths(split_roots, class_names)
        if args.max_samples > 0:
            cap = args.max_samples * len(class_names)
            file_paths = file_paths[:cap]
            labels = labels[:cap]
        args.out_npy.parent.mkdir(parents=True, exist_ok=True)
        n_samples = len(file_paths)
        if args.image_size % 8 != 0:
            raise ValueError(f"image_size must be divisible by 8 for SD-VAE. Got {args.image_size}")
        d = 4 * (args.image_size // 8) * (args.image_size // 8)
        vecs = open_memmap(str(args.out_npy), mode="w+", dtype=np.float32, shape=(n_samples, d))
        i = 0
        with torch.no_grad():
            while i < n_samples:
                batch_paths = file_paths[i : i + args.vae_batch_size]
                batch_chw = []
                for img_path in batch_paths:
                    with Image.open(img_path) as img:
                        chw = resize_center_crop_rgb(img, args.image_size)
                    batch_chw.append(chw * 2.0 - 1.0)
                x = torch.from_numpy(np.stack(batch_chw, axis=0)).to(device=device, dtype=torch.float32)
                posterior = vae.encode(x).latent_dist
                latents = posterior.mode() * 0.18215
                flat_batch = latents.detach().cpu().numpy().reshape(len(batch_paths), -1).astype(np.float32, copy=False)
                vecs[i : i + len(batch_paths)] = flat_batch
                i += len(batch_paths)
        if args.save_labels:
            label_path = args.out_npy.with_name(f"{args.out_npy.stem}_labels.npy")
            np.save(str(label_path), np.asarray(labels, dtype=np.int64))
            print(f"Saved labels to: {label_path}")
        print(f"Saved merged npy: {args.out_npy}")

    print("Conversion done.")
    print(f"Output dir: {args.out_dir}")
    print(f"Spec: {args.spec}")
    print(f"Split: {args.split}")
    print(f"Image size: {args.image_size}")
    print(f"VAE model: {model_id}")
    for class_name, n_samples, mean, std in class_summary:
        print(f"- {class_name}: samples={n_samples}, mean={mean:.6f}, std={std:.6f}")


if __name__ == "__main__":
    main()
