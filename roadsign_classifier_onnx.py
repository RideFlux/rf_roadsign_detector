import torch
import torch.nn as nn
from roadsign_classifier import Classifier


device = torch.device('cpu')
model = Classifier(in_channels=3, num_classes=9).to(device)
checkpoint = torch.load('classfier_checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
class ModelWithArgmax(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        predicted_class = torch.argmax(logits, dim=1).to(torch.int32)
        return predicted_class
    
class ModelWithSoftmax(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.model(x)
        probs = self.softmax(logits)      # softmax 확률
        max_prob, max_idx = torch.max(probs, dim=1)  # 최대값과 index
        print(max_prob, type(max_prob), max_prob.dtype)
        print(max_idx, type(max_idx), max_idx.dtype)
        return max_idx, max_prob

final_model = ModelWithSoftmax(model)
final_model.eval()
dummy_input = torch.randn(1, 3, 64, 32, dtype=torch.float32)
onnx_file_path = "roadsign_classifier.onnx"
torch.onnx.export(
    final_model,
    dummy_input,
    onnx_file_path,
    input_names=['points'],
    output_names=['pred_class', 'pred_score'],
)