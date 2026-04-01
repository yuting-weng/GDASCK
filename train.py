from __future__ import annotations

import argparse

import torch

from src.data.dataset import create_data_bundle, create_paired_data_bundle
from src.eval.export_dataset import export_generated_dataset
from src.eval.smoke import run_smoke
from src.trainers.gan_trainer import train_gan, train_gan_three_stage
from src.trainers.quality_trainer import train_and_score_quality
from src.utils import adapt_hidden_dims, ensure_runtime_dirs, load_config, save_json, select_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CPC-GAN 训练入口")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    sp = p.add_subparsers(dest="command", required=True)

    def _add_module_toggle_args(subp: argparse.ArgumentParser) -> None:
        subp.add_argument("--enable_species_bounds", dest="enable_species_bounds", action="store_true", default=None)
        subp.add_argument("--disable_species_bounds", dest="enable_species_bounds", action="store_false")
        subp.add_argument("--enable_minibatch_disc", dest="enable_minibatch_disc", action="store_true", default=None)
        subp.add_argument("--disable_minibatch_disc", dest="enable_minibatch_disc", action="store_false")
        subp.add_argument("--enable_condition_encoder", dest="enable_condition_encoder", action="store_true", default=None)
        subp.add_argument("--disable_condition_encoder", dest="enable_condition_encoder", action="store_false")

    p_gan = sp.add_parser("train_gan")
    p_gan.add_argument("--subset_size", type=int, default=None)
    _add_module_toggle_args(p_gan)

    p_q = sp.add_parser("train_quality_dnn")
    p_q.add_argument("--mode", type=str, default=None, choices=["classifier", "error_regression", "hybrid"])
    p_q.add_argument("--subset_size", type=int, default=None)

    p_s = sp.add_parser("smoke_test")
    p_s.add_argument("--subset_size", type=int, default=2048)
    _add_module_toggle_args(p_s)

    p_e = sp.add_parser("generate_dataset")
    p_e.add_argument("--gan_checkpoint", type=str, default=None)
    p_e.add_argument("--transform_stats", type=str, default=None)
    p_e.add_argument("--target_size", type=int, default=None)
    p_e.add_argument("--subset_size", type=int, default=None)
    _add_module_toggle_args(p_e)
    return p


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return select_device(prefer_cuda=True)


def _apply_module_overrides(cfg: dict, args) -> None:
    train_cfg = cfg.setdefault("train", {})
    phy_cfg = train_cfg.setdefault("physics_species_bounds", {})
    model_cfg = cfg.setdefault("model", {})
    critic_cfg = model_cfg.setdefault("critic", {})
    mb_cfg = critic_cfg.setdefault("minibatch_discrimination", {})
    gen_cfg = model_cfg.setdefault("generator", {})
    cond_cfg = gen_cfg.setdefault("condition_encoder", {})
    if getattr(args, "enable_species_bounds", None) is not None:
        phy_cfg["enabled"] = bool(args.enable_species_bounds)
    if getattr(args, "enable_minibatch_disc", None) is not None:
        mb_cfg["enabled"] = bool(args.enable_minibatch_disc)
    if getattr(args, "enable_condition_encoder", None) is not None:
        cond_cfg["enabled"] = bool(args.enable_condition_encoder)


def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    _apply_module_overrides(cfg, args)
    set_seed(int(cfg["seed"]))
    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但当前不可用。")
    print(f"[INFO] device={device} cuda_available={torch.cuda.is_available()}")

    runtime = ensure_runtime_dirs(cfg["output_root"], args.command)

    if args.command == "smoke_test":
        cfg["data"]["subset_size"] = args.subset_size
        summary = run_smoke(cfg, runtime.run_dir, device)
        print("[SMOKE]", summary)
        return

    if args.subset_size is not None:
        cfg["data"]["subset_size"] = args.subset_size

    if args.command == "generate_dataset":
        if args.target_size is not None:
            cfg.setdefault("generate", {})
            cfg["generate"]["target_size"] = int(args.target_size)
        summary = export_generated_dataset(
            config=cfg,
            run_dir=runtime.run_dir,
            device=device,
            gan_checkpoint=args.gan_checkpoint,
            transform_stats_path=args.transform_stats,
        )
        save_json(summary, runtime.run_dir / "generate_dataset_summary.json")
        print("[GENERATE]", summary)
        return

    bundle = create_data_bundle(
        npy_path=cfg["data"]["npy_path"],
        batch_size=int(cfg["data"]["batch_size"]),
        val_ratio=float(cfg["data"]["val_ratio"]),
        seed=int(cfg["seed"]),
        num_workers=int(cfg["data"].get("num_workers", 0)),
        subset_size=cfg["data"].get("subset_size"),
        use_bct=bool(cfg["transform"]["use_bct"]),
        bct_epsilon=float(cfg["transform"]["bct_epsilon"]),
        standardize=bool(cfg["transform"]["standardize"]),
        disable_input_dim0_bct=bool(cfg["transform"].get("disable_input_dim0_bct", False)),
    )
    cfg["model"] = adapt_hidden_dims(cfg["model"], bundle.feature_dim)
    save_json(cfg, runtime.run_dir / "config_snapshot.json")
    bundle.transform.save(runtime.run_dir / "transform_stats.npz")

    condition_dim = int(cfg["data"].get("condition_dim", 0))
    if args.command == "train_gan":
        use_three_stage = bool(cfg["train"].get("use_three_stage", True))
        species_min_raw = bundle.train_raw[:, 1:].min(axis=0) if bundle.train_raw.shape[1] > 1 else None
        species_max_raw = bundle.train_raw[:, 1:].max(axis=0) if bundle.train_raw.shape[1] > 1 else None
        if use_three_stage:
            quality_cfg = cfg.get("quality", {})
            paired_bundle = create_paired_data_bundle(
                input_npy_path=str(quality_cfg.get("regression_input_path")),
                target_npy_path=str(quality_cfg.get("regression_target_path")),
                batch_size=int(cfg["data"]["batch_size"]),
                val_ratio=float(cfg["data"]["val_ratio"]),
                seed=int(cfg["seed"]),
                num_workers=int(cfg["data"].get("num_workers", 0)),
                subset_size=cfg["data"].get("subset_size"),
                use_bct=bool(cfg["transform"]["use_bct"]),
                bct_epsilon=float(cfg["transform"]["bct_epsilon"]),
                standardize=bool(cfg["transform"]["standardize"]),
                disable_input_dim0_bct=bool(cfg["transform"].get("disable_input_dim0_bct", False)),
            )
            print(
                "[TRANSFORM] gan_input_use_bct="
                f"{bundle.transform.use_bct} dim0_bct_enabled={bool(bundle.transform.mask[0])} "
                f"paired_input_dim0_bct_enabled={bool(paired_bundle.input_transform.mask[0])} "
                f"paired_target_use_bct={paired_bundle.target_transform.use_bct}"
            )
            paired_bundle.input_transform.save(runtime.run_dir / "reg_input_transform_stats.npz")
            paired_bundle.target_transform.save(runtime.run_dir / "reg_target_transform_stats.npz")
            _, _, metrics = train_gan_three_stage(
                train_loader=bundle.train_loader,
                paired_loader=paired_bundle.train_loader,
                transform=bundle.transform,
                target_transform=paired_bundle.target_transform,
                feature_dim=bundle.feature_dim,
                target_dim=paired_bundle.target_dim,
                model_cfg=cfg["model"],
                optim_cfg=cfg["optim"],
                train_cfg=cfg["train"],
                quality_cfg=quality_cfg,
                output_dir=runtime.run_dir,
                device=device,
                condition_dim=condition_dim,
                species_min_raw=species_min_raw,
                species_max_raw=species_max_raw,
            )
        else:
            _, _, metrics = train_gan(
                train_loader=bundle.train_loader,
                transform=bundle.transform,
                feature_dim=bundle.feature_dim,
                model_cfg=cfg["model"],
                optim_cfg=cfg["optim"],
                train_cfg=cfg["train"],
                output_dir=runtime.run_dir,
                device=device,
                condition_dim=condition_dim,
                species_min_raw=species_min_raw,
                species_max_raw=species_max_raw,
            )
        print("[GAN]", metrics)
        return

    if args.command == "train_quality_dnn":
        quality_cfg = cfg.get("quality", {})
        mode = args.mode or str(quality_cfg.get("default_mode", "hybrid"))
        paired_bundle = create_paired_data_bundle(
            input_npy_path=str(quality_cfg.get("regression_input_path")),
            target_npy_path=str(quality_cfg.get("regression_target_path")),
            batch_size=int(cfg["data"]["batch_size"]),
            val_ratio=float(cfg["data"]["val_ratio"]),
            seed=int(cfg["seed"]),
            num_workers=int(cfg["data"].get("num_workers", 0)),
            subset_size=cfg["data"].get("subset_size"),
            use_bct=bool(cfg["transform"]["use_bct"]),
            bct_epsilon=float(cfg["transform"]["bct_epsilon"]),
            standardize=bool(cfg["transform"]["standardize"]),
            disable_input_dim0_bct=bool(cfg["transform"].get("disable_input_dim0_bct", False)),
        )
        paired_bundle.input_transform.save(runtime.run_dir / "reg_input_transform_stats.npz")
        paired_bundle.target_transform.save(runtime.run_dir / "reg_target_transform_stats.npz")
        g, _, _ = train_gan(
            train_loader=bundle.train_loader,
            transform=bundle.transform,
            feature_dim=bundle.feature_dim,
            model_cfg=cfg["model"],
            optim_cfg=cfg["optim"],
            train_cfg={**cfg["train"], "epochs_gan": 1, "n_critic": 1},
            output_dir=runtime.run_dir,
            device=device,
            condition_dim=condition_dim,
        )
        metrics = train_and_score_quality(
            gan_loader=bundle.train_loader,
            paired_loader=paired_bundle.train_loader,
            generator=g,
            feature_dim=bundle.feature_dim,
            target_dim=paired_bundle.target_dim,
            model_cfg=cfg["model"],
            optim_cfg=cfg["optim"],
            train_cfg=cfg["train"],
            quality_cfg=quality_cfg,
            output_dir=runtime.run_dir,
            device=device,
            mode=mode,
            condition_dim=condition_dim,
            gan_transform=bundle.transform,
            target_transform=paired_bundle.target_transform,
        )
        print("[QUALITY]", metrics)
        return


if __name__ == "__main__":
    main()
