import os
import argparse
import numpy as np
import timm
import wandb
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from src.perspective_data import FeaturesDataset
from src.models import LinearModel
from src.utils import binary_accuracy, get_args_parser, get_transform_wo_crop
import json
import csv


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder.
    """
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (original_tuple + (path,))


def extract_features(model, data_loader, device, return_path=False):
    model.eval()
    features = []
    labels_list = []
    img_path_list = []
    
    for data in tqdm(data_loader):
        if return_path:
            images, labels, img_paths = data
            img_path_list += img_paths
        else:
            images, labels = data
            
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            preds = model(images)
            features.append(preds.cpu())
            labels_list.append(labels.cpu())
            
    features = torch.cat(features)
    labels = torch.cat(labels_list).squeeze()
    
    if not return_path:
        return features, labels
    else:
        return features, labels, img_path_list


def train_linear_probe(model, train_loader, test_loader, criterion, optimizer, device, args):
    best_acc_test = 0
    best_acc_train = 0

    for epoch in tqdm(range(args.epochs)):
        # --- Training Step ---
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
        
        train_acc = sum(epoch_acc) / float(len(epoch_acc))
        train_loss = sum(epoch_loss) / float(len(epoch_loss))

        # --- Evaluation Step ---
        with torch.no_grad():
            test_acc, test_loss, test_records = evaluate_linear_probe(model, test_loader, criterion, device, True)

            # Save best model based on Test Accuracy
            if test_acc > best_acc_test:
                best_acc_test = test_acc
                best_acc_train = train_acc
                
                # Save predictions CSV
                file_name = f'{args.model_name}_{"preds_depth" if args.task == "depth" else "preds"}_lp.csv'
                os.makedirs('./logs/linear_probe_preds', exist_ok=True)
                
                with open(f'./logs/linear_probe_preds/{file_name}', 'w') as f:
                    header = ['path', 'pred', 'label']
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(test_records.tolist())
                    
                # Save Model Checkpoint
                torch.save(model, f'./logs/linear_probe_preds/{args.model_name}_{args.task}.ckpt')

        if args.wandb:
            wandb.log({
                'train_acc': train_acc, 
                'train_loss': train_loss,
                'test_acc': test_acc,
                'test_loss': test_loss
            })
            
    return best_acc_train, best_acc_test


def evaluate_linear_probe(model, data_loader, criterion, device, return_record):
    model.eval()
    epoch_loss = []
    preds_list = []
    labels_list = []
    preds_list_logits = []
    img_path_list = []

    for i, batch in enumerate(data_loader):
        if return_record:
            features, labels, img_path = batch
            img_path_list += img_path
        else:
            features, labels = batch
        
        features = features.to(device)
        labels = labels.float().to(device)
        labels = torch.unsqueeze(labels, 1)
        
        preds = model(features)
        loss = criterion(preds, labels)
        
        preds_list_logits.append(preds)
        labels_list.append(labels)
        epoch_loss.append(loss.item())
        
        if return_record:
            preds_list.append(preds)

    preds = torch.cat(preds_list_logits).squeeze().cpu()
    labels = torch.cat(labels_list).squeeze().cpu()
    epoch_acc = binary_accuracy(preds, labels)

    if return_record:
        img_path_record = np.array(img_path_list).squeeze().T
        preds_record = torch.cat(preds_list).cpu().numpy().squeeze().T
        labels_record = torch.cat(labels_list).cpu().numpy().squeeze().T
        records = np.vstack([img_path_record, preds_record, labels_record]).T
        return epoch_acc, sum(epoch_loss)/float(len(epoch_loss)), records
    
    return epoch_acc, sum(epoch_loss)/float(len(epoch_loss))


def run_extract_features(args):
    device = torch.device(f'cuda:{args.gpu_id}')
    print(args.model_name)
    model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
    
    data_config = timm.data.resolve_model_data_config(model)
    transform = get_transform_wo_crop(data_config)

    # Train Data
    if not args.flip:
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=transform)
    else:
        train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train_flip'), transform=transform)
    
    # Test Data
    test_dataset = ImageFolderWithPaths(os.path.join(args.data_dir, 'test'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.extract_batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.extract_batch_size, num_workers=args.num_workers, pin_memory=True)
    
    model = model.to(device)
    
    # Extract
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels, test_img_paths = extract_features(model, test_loader, device, return_path=True)

    return train_features, train_labels, test_features, test_labels, test_img_paths


def run_linear_probe(args):
    if args.wandb:
        wandb_run = wandb.init(
            project='gs-perception-linear-probe', 
            config={                   
                "learning_rate": args.learning_rate,
                "architecture": args.model_name,
                "epochs": args.epochs,
            }
        )
    
    device = torch.device(f'cuda:{args.gpu_id}')

    # Extract Features
    train_features, train_labels, test_features, test_labels, test_img_paths = run_extract_features(args)
        
    # Create Feature Datasets
    train_feat_dataset = FeaturesDataset(train_features, train_labels)
    test_feat_dataset = FeaturesDataset(test_features, test_labels, test_img_paths)
    
    train_feat_loader = DataLoader(
        train_feat_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True, drop_last=True
    )
    test_feat_loader = DataLoader(
        test_feat_dataset, batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    criterion = torch.nn.BCEWithLogitsLoss()
    linear_model = LinearModel(train_features.shape[-1], args.num_classes, args.dropout_rate)
    optimizer = torch.optim.AdamW(
        linear_model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay, amsgrad=False
    )
    
    linear_model = linear_model.to(device)
    
    # Train Probe
    best_acc_train, best_acc_test = train_linear_probe(
        linear_model, train_feat_loader, test_feat_loader, 
        criterion, optimizer, device, args
    )
    
    print("Best acc test", best_acc_test)
    print("Best acc train", best_acc_train)

    # Logging Results
    log_file = f'logs/{"perspective" if args.task == "perspective" else "depth"}_results.json'
        
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}
    else:
        results = {}
        
    # Saving only Train and Test
    results[args.model_name] = [float(best_acc_train), float(best_acc_test)]
    
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    if args.wandb:
        wandb_run.finish()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    run_linear_probe(args)