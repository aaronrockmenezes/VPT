import os
import argparse
import numpy as np
import timm
import wandb
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from accelerate import Accelerator
from accelerate.utils import gather_object
import json
import csv

from src.perspective_data import PerspectiveDataset, FeaturesDataset
from src.models import LinearModel, LinearModelMulti
from src.utils import binary_accuracy, accuracy, CosineAnnealingWithWarmup, get_args_parser, get_transform_wo_crop


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder.
    """
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (original_tuple + (path,))


def parse_label_map(label_map_str):
    """
    Parse a label map string like 'no:0,yes:1' or 'left:0,right:1'
    into a dict: {'no': 0, 'yes': 1}
    """
    label_map = {}
    for pair in label_map_str.split(','):
        key, val = pair.strip().split(':')
        label_map[key.strip().lower()] = int(val.strip())
    return label_map


def apply_label_map(dataset, label_map):
    """
    Remaps dataset targets in-place according to label_map.
    Raises if any folder name in the dataset is not in label_map.
    """
    for cls in dataset.classes:
        if cls.lower() not in label_map:
            raise ValueError(f"Folder '{cls}' not found in label_map. Expected one of: {list(label_map.keys())}")

    # Override class_to_idx with our explicit mapping
    dataset.class_to_idx = {cls: label_map[cls.lower()] for cls in dataset.classes}

    # Remap targets and samples/imgs
    dataset.targets = [dataset.class_to_idx[dataset.classes[t]] for t in dataset.targets]
    dataset.samples = [(path, dataset.class_to_idx[dataset.classes[orig_label]]) for path, orig_label in dataset.imgs]
    dataset.imgs = dataset.samples


def extract_features(model, data_loader, accelerator, return_path=False):
    model.eval()
    features_list = []
    labels_list = []
    # Stores (path, label) tuples per sample so they stay in sync through gather
    path_label_list = []

    if accelerator.is_main_process:
        iterator = tqdm(data_loader)
    else:
        iterator = data_loader

    for batch in iterator:
        if return_path:
            images, labels, paths = batch
            # Zip path+label together so they travel through gather_object as a unit
            path_label_list.extend(zip(paths, labels.tolist()))
        else:
            images, labels = batch

        with torch.no_grad():
            preds = model(images)

        all_preds, all_labels = accelerator.gather_for_metrics((preds, labels))

        features_list.append(all_preds.cpu())
        labels_list.append(all_labels.cpu())

    features = torch.cat(features_list)
    labels = torch.cat(labels_list).squeeze()

    if return_path:
        # gather_object preserves insertion order per rank — path and label stay paired
        all_path_labels = gather_object(path_label_list)

        # Trim to match gathered feature count (handles uneven last batch)
        all_path_labels = all_path_labels[:len(features)]
        all_paths = [pl[0] for pl in all_path_labels]
        all_labels_from_paths = [pl[1] for pl in all_path_labels]

        return features, labels, all_paths, all_labels_from_paths

    return features, labels


def train_linear_probe(model, train_loader, test_features, test_labels, criterion, optimizer, device, args):
    """
    Train for args.epochs, return best-epoch metrics + predictions + model state.
    No file I/O — caller handles saving.
    """
    best_test_acc = 0
    best_train_acc = 0
    best_preds = None
    best_state_dict = None

    # Pre-load test data to GPU once
    test_features_gpu = test_features.to(device)
    test_labels_gpu = test_labels.float().to(device).unsqueeze(1)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        epoch_acc = []
        epoch_loss = []
        for i, batch in enumerate(train_loader):
            features, labels = batch
            features = features.to(device)
            labels = labels.float().to(device)
            labels = torch.unsqueeze(labels, 1)

            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            acc = binary_accuracy(preds, labels)
            epoch_acc.append(acc)
            epoch_loss.append(loss.item())

        # Eval on test set
        with torch.no_grad():
            model.eval()
            preds = model(test_features_gpu)
            test_acc = binary_accuracy(preds.squeeze().cpu(), test_labels.float())
            train_acc = sum(epoch_acc) / float(len(epoch_acc))

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_train_acc = train_acc
                best_preds = preds.cpu().numpy().squeeze()
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if args.wandb:
            wandb.log({
                'train_acc': train_acc,
                'train_loss': sum(epoch_loss) / float(len(epoch_loss)),
                'test_acc': test_acc,
            })

    return {
        'train_acc': float(best_train_acc),
        'test_acc': float(best_test_acc),
        'preds': best_preds,
        'state_dict': best_state_dict,
    }


def run_extract_features(args, accelerator, label_map):
    if accelerator.is_main_process:
        print(args.model_name)

    model = timm.create_model(args.model_name, pretrained=True, num_classes=0)

    data_config = timm.data.resolve_model_data_config(model)
    transform = get_transform_wo_crop(data_config)

    if args.task == 'depth':
        train_dir = 'train_depth_flip' if args.flip else 'train_depth'
        test_dir = 'test_depth'
    else:
        # Works for both 'perspective' and 'vpt2' (and any future binary task)
        train_dir = 'train_flip' if args.flip else 'train'
        test_dir = 'test'

    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, train_dir), transform=transform)
    test_dataset = ImageFolderWithPaths(os.path.join(args.data_dir, test_dir), transform=transform)

    # Apply explicit label mapping — overrides ImageFolder's default alphabetical assignment
    apply_label_map(train_dataset, label_map)
    apply_label_map(test_dataset, label_map)

    train_loader = DataLoader(train_dataset, batch_size=args.extract_batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.extract_batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    model, train_loader, test_loader = accelerator.prepare(
        model, train_loader, test_loader
    )

    train_features, train_labels = extract_features(model, train_loader, accelerator)
    test_features, test_labels, test_img_paths, test_path_labels = extract_features(model, test_loader, accelerator, return_path=True)

    return train_features, train_labels, test_features, test_labels, test_img_paths, test_path_labels


def save_preds_csv(csv_path, preds_np, test_img_paths, test_path_labels):
    """Save per-sample predictions to CSV."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'pred', 'label'])
        if test_img_paths is not None:
            rows = [
                [path, float(pred), int(lbl)]
                for path, pred, lbl in zip(test_img_paths, preds_np, test_path_labels)
            ]
            writer.writerows(rows)


def run_linear_probe(args):
    # ---- Resolve label map ----
    if args.label_map:
        label_map = parse_label_map(args.label_map)
    else:
        label_map = {'no': 0, 'yes': 1}
    print(f"Label map: {label_map}")

    accelerator = Accelerator()

    # 1. Extract Features ONCE (Multi-GPU)
    data = run_extract_features(args, accelerator, label_map)

    accelerator.wait_for_everyone()

    # 2. Train Linear Probes (Main Process Only)
    if not accelerator.is_main_process:
        return

    train_features, train_labels, test_features, test_labels, test_img_paths, test_path_labels = data

    if args.wandb:
        wandb_run = wandb.init(project='gs-perception-linear-probe',
                               config={
                                   "learning_rate": args.learning_rate,
                                   "architecture": args.model_name,
                                   "epochs": args.epochs,
                                   "num_runs": args.num_runs,
                               })

    device = accelerator.device
    criterion = torch.nn.BCEWithLogitsLoss()

    train_feat_dataset = FeaturesDataset(train_features, train_labels)
    train_feat_loader = DataLoader(train_feat_dataset, batch_size=args.batch_size,
                                   num_workers=args.num_workers, shuffle=True, drop_last=True)

    # Output dirs
    results_dir = os.path.join(args.output_dir, 'results')
    ckpts_dir = os.path.join(args.output_dir, 'ckpts')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpts_dir, exist_ok=True)

    # ---- Multi-run loop ----
    all_run_results = []
    best_run_idx = -1
    best_run_test_acc = -1

    for run_idx in range(1, args.num_runs + 1):
        print(f"\n{'='*40}")
        print(f"  Run {run_idx} / {args.num_runs}")
        print(f"{'='*40}")

        # Fresh linear model each run
        linear_model = LinearModel(train_features.shape[-1], args.num_classes, args.dropout_rate)
        linear_model = linear_model.to(device)
        optimizer = torch.optim.AdamW(linear_model.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay, amsgrad=False)

        run_result = train_linear_probe(
            linear_model, train_feat_loader, test_features, test_labels,
            criterion, optimizer, device, args
        )

        print(f"  Run {run_idx}: train_acc={run_result['train_acc']:.4f}  test_acc={run_result['test_acc']:.4f}")

        # Save per-run predictions CSV
        csv_path = os.path.join(results_dir, f'{args.model_name}_run{run_idx}_preds.csv')
        save_preds_csv(csv_path, run_result['preds'], test_img_paths, test_path_labels)

        all_run_results.append({
            'run': run_idx,
            'train_acc': run_result['train_acc'],
            'test_acc': run_result['test_acc'],
        })

        # Track best run for checkpoint
        if run_result['test_acc'] > best_run_test_acc:
            best_run_test_acc = run_result['test_acc']
            best_run_idx = run_idx
            best_state_dict = run_result['state_dict']

    # ---- Save best checkpoint ----
    # Reconstruct model, load best weights, save
    best_model = LinearModel(train_features.shape[-1], args.num_classes, args.dropout_rate)
    best_model.load_state_dict(best_state_dict)
    ckpt_path = os.path.join(ckpts_dir, f'{args.model_name}.ckpt')
    torch.save(best_model, ckpt_path)
    print(f"\nSaved best checkpoint (run {best_run_idx}, test_acc={best_run_test_acc:.4f}) -> {ckpt_path}")

    # ---- Save combined JSON ----
    avg_test_acc = float(np.mean([r['test_acc'] for r in all_run_results]))
    avg_train_acc = float(np.mean([r['train_acc'] for r in all_run_results]))

    json_data = {
        'model': args.model_name,
        'task': args.task,
        'runs': all_run_results,
        'avg_train_acc': round(avg_train_acc, 4),
        'avg_test_acc': round(avg_test_acc, 4),
        'best_run': best_run_idx,
        'best_test_acc': round(best_run_test_acc, 4),
    }
    json_path = os.path.join(results_dir, f'{args.model_name}.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"Saved results -> {json_path}")
    print(f"Avg test acc across {args.num_runs} runs: {avg_test_acc:.4f}")

    if args.wandb:
        wandb_run.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument('--label_map', type=str, default=None,
                        help='Label mapping as "key1:val1,key2:val2" (e.g. "no:0,yes:1" or "left:0,right:1"). '
                             'Defaults to no:0,yes:1 if not provided.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Root output directory. Saves to output_dir/{results,ckpts,logs}/')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of independent linear probe runs (default: 3)')
    args = parser.parse_args()
    run_linear_probe(args)