"""Evaluate the classifier on saved crops (GT and predicted crops as listed)."""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.plots import save_confusion_matrix
from roadsign_classifier import ImageClassificationDataset, Classifier


@torch.inference_mode()
def evaluate_classifier(model, dataloader, device):
    model.eval()
    matrix = np.zeros((9, 9), dtype=np.int64)
    for images, labels in tqdm(dataloader, desc='Classifier evaluation'):
        predictions = model(images.to(device)).argmax(dim=1).cpu().numpy()
        labels = labels.numpy()
        matrix += np.bincount(labels * 9 + predictions, minlength=81).reshape(9, 9)
    if not matrix.sum():
        raise ValueError('The classifier evaluation dataset is empty')
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='classifier_261112_checkpoints/best_model.pth')
    parser.add_argument('--labels', default='Dataset/test_classification_261112.txt')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', default='confusion_matrix_classifier_261112.png')
    args = parser.parse_args()
    device_name = args.device
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    dataset = ImageClassificationDataset(image_dir='test', label_file=args.labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = Classifier(in_channels=3, num_classes=9).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f"Device: {device}, test samples: {len(dataset)}, checkpoint epoch: {checkpoint.get('epoch')}")
    matrix = evaluate_classifier(model, dataloader, device)
    tp = matrix.diagonal()
    print('tp:', tp)
    print('fp:', matrix.sum(axis=0) - tp)
    print('fn:', matrix.sum(axis=1) - tp)
    print(matrix)
    print(f'Accuracy: {tp.sum() / matrix.sum():.6%} ({tp.sum()}/{matrix.sum()})')
    names = [f'Class_{i + 1}' for i in range(8)] + ['No object']
    save_confusion_matrix(matrix, names, args.output)
    print(f'Confusion matrix: {args.output}')


if __name__ == '__main__':
    main()
