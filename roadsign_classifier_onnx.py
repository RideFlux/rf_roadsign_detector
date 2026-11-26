"""Export the final roadsign classifier to ONNX."""

import argparse

import torch
import torch.nn as nn

from roadsign_classifier import Classifier


class ModelWithSoftmax(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        probabilities = self.model(x).softmax(dim=1)
        score, class_id = probabilities.max(dim=1)
        return class_id, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='classifier_261112_checkpoints/best_model.pth')
    parser.add_argument('--output', default='roadsign_classifier_ver219.onnx')
    args = parser.parse_args()
    model = Classifier(in_channels=3, num_classes=9)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    final_model = ModelWithSoftmax(model).eval()
    torch.onnx.export(
        final_model, torch.randn(1, 3, 64, 32), args.output,
        input_names=['points'], output_names=['pred_class', 'pred_score'],
    )
    print(f'Classifier ONNX: {args.output}')


if __name__ == '__main__':
    main()
