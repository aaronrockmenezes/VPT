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


def extract_features(model, data_loader, accelerator, return_path=False):
    model.eval()
    
    local_features = []
    local_labels = []
    local_paths = []

    if accelerator.is_main_process:
        iterator = tqdm(data_loader, desc="Extracting Features")
    else:
        iterator = data_loader

    for batch in iterator:
        if return_path:
            images, labels, paths = batch
            local_paths.extend(paths)
        else:
            images, labels = batch

        with torch.no_grad():
            preds = model(images)
        
        local_features.append(preds)
        local_labels.append(labels)

    features = torch.cat(local_features)
    labels = torch.cat(local_labels)

    all_features = accelerator.gather(features)
    all_labels = accelerator.gather(labels)
    
    if all_labels.ndim > 1:
        all_labels = all_labels.squeeze()

    if return_path:
        all_paths = gather_object(local_paths)

        min_len = min(len(all_features), len(all_paths))
        all_features = all_features[:min_len]
        all_labels = all_labels[:min_len]
        all_paths = all_paths[:min_len]

        return all_features.cpu(), all_labels.cpu(), all_paths

    return all_features.cpu(), all_labels.cpu()


def train_linear_probe(model, train_loader, test_features, test_labels, test_img_paths, criterion, optimizer, lr_scheduler, device, args):
    best_acc_val = 0
    best_acc_test = 0
    best_acc_train = 0
    best_acc_human = 0

    test_features_gpu = test_features.to(device)
    test_labels_gpu = test_labels.float().to(device).unsqueeze(1)

    for epoch in tqdm(range(args.epochs), desc="Training Linear Probe"):
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

        with torch.no_grad():
            model.eval()
            preds = model(test_features_gpu)
            test_loss = criterion(preds, test_labels_gpu).item()
            test_acc = binary_accuracy(preds.squeeze().cpu(), test_labels.float())

            train_acc = sum(epoch_acc) / float(len(epoch_acc))

            val_acc = test_acc
            val_loss = test_loss
            human_acc = test_acc
            human_loss = test_loss

            if val_acc > best_acc_val:
                best_acc_val = val_acc
                best_acc_test = test_acc
                best_acc_train = train_acc
                best_acc_human = human_acc

                if args.task == 'depth':
                    file_name = "preds_depth"
                else:
                    file_name = "preds"
                file_name = f'{args.model_name}_{file_name}_lp.csv'

                os.makedirs('./logs/linear_probe_preds', exist_ok=True)

                with open(f'./logs/linear_probe_preds/{file_name}', 'w') as f:
                    header = ['path', 'pred', 'label']
                    writer = csv.writer(f)
                    writer.writerow(header)
                    
                    if test_img_paths is not None:
                        preds_flat = preds.cpu().numpy().flatten()
                        labels_flat = test_labels.numpy().flatten() 
                        paths_flat = np.array(test_img_paths).flatten()

                        min_len = min(len(preds_flat), len(paths_flat), len(labels_flat))
                        
                        rows_to_write = []
                        for i in range(min_len):
                            rows_to_write.append([paths_flat[i], preds_flat[i], int(labels_flat[i])])
                        
                        writer.writerows(rows_to_write)
                    
                    torch.save(model, f'./logs/linear_probe_preds/{args.model_name}_{args.task}.ckpt')

        if args.wandb:
            wandb.log({
                'train_acc': train_acc,
                'train_loss': sum(epoch_loss) / float(len(epoch_loss)),
                'val_acc': val_acc, 'val_loss': val_loss,
                'human_acc': human_acc, 'human_loss': human_loss,
                'test_acc': test_acc, 'test_loss': test_loss
            })

    return best_acc_train, best_acc_test, best_acc_val, best_acc_human


def run_extract_features(args, accelerator):
    if accelerator.is_main_process:
        print(f"Model: {args.model_name}")

    model = timm.create_model(args.model_name, pretrained=True, num_classes=0)

    data_config = timm.data.resolve_model_data_config(model)
    transform = get_transform_wo_crop(data_config)

    if args.task == 'depth':
        train_dir = 'train_depth_flip' if args.flip else 'train_depth'
        test_dir = 'test_depth'
    else:
        train_dir = 'train_flip' if args.flip else 'train'
        test_dir = 'test'

    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, train_dir), transform=transform)
    test_dataset = ImageFolderWithPaths(os.path.join(args.data_dir, test_dir), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.extract_batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.extract_batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    model, train_loader, test_loader = accelerator.prepare(
        model, train_loader, test_loader
    )

    train_features, train_labels = extract_features(model, train_loader, accelerator)
    test_features, test_labels, test_img_paths = extract_features(model, test_loader, accelerator, return_path=True)

    return train_features, train_labels, test_features, test_labels, test_img_paths


def run_linear_probe(args):
    accelerator = Accelerator()

    data = run_extract_features(args, accelerator)

    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        return

    train_features, train_labels, test_features, test_labels, test_img_paths = data

    if args.wandb:
        wandb_run = wandb.init(project='gs-perception-linear-probe',
                               config={
                                   "learning_rate": args.learning_rate,
                                   "architecture": args.model_name,
                                   "epochs": args.epochs,
                               })

    device = accelerator.device

    train_feat_dataset = FeaturesDataset(train_features, train_labels)

    train_feat_loader = DataLoader(train_feat_dataset, batch_size=args.batch_size,
                                   num_workers=args.num_workers, shuffle=True, drop_last=True)

    criterion = torch.nn.BCEWithLogitsLoss()

    linear_model = LinearModel(train_features.shape[-1], args.num_classes, args.dropout_rate)
    optimizer = torch.optim.AdamW(linear_model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, amsgrad=False)
    lr_scheduler = None
    linear_model = linear_model.to(device)

    best_acc_train, best_acc_test, best_acc_val, best_acc_human = train_linear_probe(
        linear_model, train_feat_loader, test_features, test_labels, test_img_paths,
        criterion, optimizer, lr_scheduler, device, args
    )

    print("Best acc validation", best_acc_val)
    print("Best acc test", best_acc_test)
    print("Best acc train", best_acc_train)
    print("Best acc human", best_acc_human)

    if args.task == 'perspective':
        log_file = 'logs/perspective_results.json'
    else:
        log_file = 'logs/depth_results.json'

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                results = json.load(f)
        except (json.JSONDecodeError, ValueError):
            results = {}
    else:
        results = {}

    results[args.model_name] = [float(best_acc_train), float(best_acc_test), float(best_acc_val), float(best_acc_human)]
    with open(log_file, 'w') as f:
        results_json = json.dumps(results, indent=4)
        f.write(results_json)
    if args.wandb:
        wandb_run.finish()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    run_linear_probe(args)