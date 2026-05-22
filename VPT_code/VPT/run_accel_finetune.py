import argparse
import csv
import json
import os

import numpy as np
import timm
import torch
import torchvision.datasets as datasets
import wandb
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import binary_accuracy, get_args_parser, get_transform_wo_crop


class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder returning image path."""

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, self.imgs[index][0]


def parse_label_map(label_map_str):
    """Parse label map string like ``no:0,yes:1``."""
    label_map = {}
    for pair in label_map_str.split(","):
        key, val = pair.strip().split(":")
        label_map[key.strip().lower()] = int(val.strip())
    return label_map


def apply_label_map(dataset, label_map):
    """Remap ImageFolder labels with explicit class mapping."""
    for cls in dataset.classes:
        if cls.lower() not in label_map:
            raise ValueError(
                f"Folder '{cls}' not found in label_map. Expected one of: {list(label_map.keys())}"
            )
    dataset.class_to_idx = {cls: label_map[cls.lower()] for cls in dataset.classes}
    dataset.targets = [dataset.class_to_idx[dataset.classes[t]] for t in dataset.targets]
    dataset.samples = [
        (path, dataset.class_to_idx[dataset.classes[orig_label]])
        for path, orig_label in dataset.imgs
    ]
    dataset.imgs = dataset.samples


def split_dirs(args):
    """Return task-specific train/test split directory names."""
    if args.task == "depth":
        train_dir = "train_depth_flip" if args.flip else "train_depth"
        test_dir = "test_depth"
    else:
        train_dir = "train_flip" if args.flip else "train"
        test_dir = "test"
    return train_dir, test_dir


def make_test_loader(args, transform, label_map, data_dir=None):
    """Create test loader for perspective-style ImageFolder data."""
    _, test_dir = split_dirs(args)
    root = data_dir or getattr(args, "test_data_dir", None) or args.data_dir
    test_dataset = ImageFolderWithPaths(os.path.join(root, test_dir), transform=transform)
    apply_label_map(test_dataset, label_map)
    return DataLoader(
        test_dataset,
        batch_size=args.extract_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

def make_loaders(args, transform, label_map):
    """Create train/test loaders for perspective-style ImageFolder data."""
    train_dir, _ = split_dirs(args)
    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, train_dir), transform=transform)
    apply_label_map(train_dataset, label_map)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=False,
    )
    test_loader = make_test_loader(args, transform, label_map)
    return train_loader, test_loader


def evaluate(model, data_loader, accelerator, criterion):
    """Evaluate binary classifier and return gathered metrics/predictions."""
    model.eval()
    dataset_len = len(data_loader.dataset)
    loss_sum = torch.tensor(0.0, device=accelerator.device)
    count_sum = torch.tensor(0.0, device=accelerator.device)
    preds_local = []
    labels_local = []
    path_label_local = []

    for images, labels, paths in data_loader:
        labels_float = labels.float().unsqueeze(1)
        with torch.no_grad():
            logits = model(images)
            loss = criterion(logits, labels_float)
        batch_n = torch.tensor(float(labels.numel()), device=accelerator.device)
        loss_sum += loss.detach() * batch_n
        count_sum += batch_n
        preds_local.append(logits.detach())
        labels_local.append(labels.detach())
        path_label_local.extend(zip(paths, labels.detach().cpu().tolist()))

    total_loss, total_count = accelerator.gather_for_metrics((loss_sum, count_sum))
    preds = torch.cat(preds_local, dim=0) if preds_local else torch.empty(0, 1, device=accelerator.device)
    labels = torch.cat(labels_local, dim=0) if labels_local else torch.empty(0, device=accelerator.device)
    all_preds, all_labels = accelerator.gather_for_metrics((preds, labels))
    all_preds = all_preds[:dataset_len]
    all_labels = all_labels[:dataset_len]
    all_path_labels = gather_object(path_label_local)[:dataset_len]

    if not accelerator.is_main_process:
        return None

    avg_loss = (total_loss.sum() / total_count.sum().clamp_min(1.0)).item()
    acc = binary_accuracy(all_preds.squeeze().cpu(), all_labels.float().cpu())
    paths = [x[0] for x in all_path_labels]
    path_labels = [x[1] for x in all_path_labels]
    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "preds": all_preds.squeeze().detach().cpu().numpy(),
        "labels": all_labels.detach().cpu().numpy(),
        "paths": paths,
        "path_labels": path_labels,
    }


def train_one_run(args, accelerator, run_idx, label_map):
    """Fine-tune one model run."""
    if accelerator.is_main_process:
        print(args.model_name)
        print(f"Run {run_idx}/{args.num_runs}")

    model = timm.create_model(args.model_name, pretrained=True, num_classes=args.num_classes)
    data_config = timm.data.resolve_model_data_config(model)
    transform = get_transform_wo_crop(data_config)
    train_loader, test_loader = make_loaders(args, transform, label_map)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        amsgrad=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.min_lr,
    )

    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )

    best = {"test_acc": -1.0, "train_acc": 0.0, "state_dict": None, "eval": None}

    for epoch in range(args.epochs):
        model.train()
        train_preds = []
        train_labels = []
        losses = []
        iterator = tqdm(train_loader, disable=not accelerator.is_main_process)
        for images, labels in iterator:
            labels_float = labels.float().unsqueeze(1)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels_float)
            accelerator.backward(loss)
            optimizer.step()
            losses.append(loss.detach())
            train_preds.append(logits.detach())
            train_labels.append(labels.detach())

        scheduler.step()
        train_preds = torch.cat(train_preds, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
        all_train_preds, all_train_labels = accelerator.gather_for_metrics((train_preds, train_labels))
        train_acc = binary_accuracy(all_train_preds.squeeze().cpu(), all_train_labels.float().cpu())
        train_loss = torch.stack(losses).mean().detach()
        all_train_loss = accelerator.gather_for_metrics(train_loss).mean().item()

        eval_result = evaluate(model, test_loader, accelerator, criterion)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            test_acc = eval_result["acc"]
            print(
                f"epoch={epoch + 1}/{args.epochs} "
                f"train_acc={train_acc:.4f} train_loss={all_train_loss:.4f} "
                f"test_acc={test_acc:.4f} test_loss={eval_result['loss']:.4f}"
            )
            if args.wandb:
                wandb.log({
                    "run": run_idx,
                    "epoch": epoch + 1,
                    "train_acc": train_acc,
                    "train_loss": all_train_loss,
                    "test_acc": test_acc,
                    "test_loss": eval_result["loss"],
                })
            if test_acc > best["test_acc"]:
                unwrapped = accelerator.unwrap_model(model)
                best = {
                    "test_acc": float(test_acc),
                    "train_acc": float(train_acc),
                    "state_dict": {k: v.detach().cpu().clone() for k, v in unwrapped.state_dict().items()},
                    "eval": eval_result,
                }

    return best if accelerator.is_main_process else None


def save_preds_csv(csv_path, preds_np, test_img_paths, test_path_labels):
    """Save per-sample logits to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "pred", "label"])
        writer.writerows(
            [path, float(pred), int(lbl)]
            for path, pred, lbl in zip(test_img_paths, preds_np, test_path_labels)
        )


def run_finetune(args):
    """Accelerate entry point."""
    label_map = parse_label_map(args.label_map) if args.label_map else {"no": 0, "yes": 1}
    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Label map: {label_map}")

    results_dir = os.path.join(args.output_dir, "results")
    ckpts_dir = os.path.join(args.output_dir, "ckpts")
    if accelerator.is_main_process:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(ckpts_dir, exist_ok=True)

    if args.eval_only:
        if not args.ckpt_path:
            raise ValueError("--eval_only requires --ckpt_path")
        model = torch.load(args.ckpt_path, map_location="cpu")
        data_config = timm.data.resolve_model_data_config(model)
        transform = get_transform_wo_crop(data_config)
        test_loader = make_test_loader(args, transform, label_map)
        criterion = torch.nn.BCEWithLogitsLoss()
        model, test_loader = accelerator.prepare(model, test_loader)
        eval_result = evaluate(model, test_loader, accelerator, criterion)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            stem = os.path.splitext(os.path.basename(args.ckpt_path))[0]
            csv_path = os.path.join(results_dir, f"{stem}_eval_preds.csv")
            save_preds_csv(csv_path, eval_result["preds"], eval_result["paths"], eval_result["path_labels"])
            summary = {
                "model": args.model_name,
                "task": args.task,
                "ckpt_path": args.ckpt_path,
                "data_dir": args.test_data_dir or args.data_dir,
                "test_acc": round(float(eval_result["acc"]), 4),
                "test_loss": round(float(eval_result["loss"]), 4),
            }
            json_path = os.path.join(results_dir, f"{stem}_eval.json")
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=4)
            print(f"Saved eval preds -> {csv_path}")
            print(f"Saved eval json -> {json_path}")
        return

    if args.wandb and accelerator.is_main_process:
        wandb.init(
            project="gs-perception-finetune",
            config={
                "learning_rate": args.learning_rate,
                "architecture": args.model_name,
                "epochs": args.epochs,
                "num_runs": args.num_runs,
            },
        )

    all_runs = []
    best_run_idx = -1
    best_acc = -1.0
    best_state = None

    for run_idx in range(1, args.num_runs + 1):
        best = train_one_run(args, accelerator, run_idx, label_map)
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            continue

        csv_path = os.path.join(results_dir, f"{args.model_name}_run{run_idx}_preds.csv")
        save_preds_csv(csv_path, best["eval"]["preds"], best["eval"]["paths"], best["eval"]["path_labels"])
        all_runs.append({
            "run": run_idx,
            "train_acc": best["train_acc"],
            "test_acc": best["test_acc"],
        })
        if best["test_acc"] > best_acc:
            best_acc = best["test_acc"]
            best_run_idx = run_idx
            best_state = best["state_dict"]

    if accelerator.is_main_process:
        if best_state is not None:
            model = timm.create_model(args.model_name, pretrained=False, num_classes=args.num_classes)
            model.load_state_dict(best_state)
            ckpt_path = os.path.join(ckpts_dir, f"{args.model_name}.ckpt")
            torch.save(model, ckpt_path)
            print(f"Saved best checkpoint -> {ckpt_path}")

        summary = {
            "model": args.model_name,
            "task": args.task,
            "runs": all_runs,
            "avg_train_acc": round(float(np.mean([r["train_acc"] for r in all_runs])), 4),
            "avg_test_acc": round(float(np.mean([r["test_acc"] for r in all_runs])), 4),
            "best_run": best_run_idx,
            "best_test_acc": round(float(best_acc), 4),
        }
        json_path = os.path.join(results_dir, f"{args.model_name}.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Saved results -> {json_path}")
        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument(
        "--label_map",
        type=str,
        default=None,
        help='Label mapping, e.g. "no:0,yes:1". Defaults to no:0,yes:1.',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root output directory. Saves to output_dir/{results,ckpts}/",
    )
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--test_data_dir", type=str, default=None,
                        help="Optional separate test dataset root.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate --ckpt_path on test split.")
    args = parser.parse_args()
    run_finetune(args)
