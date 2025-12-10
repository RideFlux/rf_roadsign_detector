import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class ImageClassificationDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None, return_file_path = False):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []
        with open(label_file, 'r') as f:
            for line in f:
                name, label = line.strip().split()
                self.samples.append((name, int(label)))
        self.return_file_path = return_file_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(filename)

        image = np.load(img_path)
        image = torch.tensor(image, dtype=torch.float32)
        if self.return_file_path:
            return image, label, filename
        else:
            return image, label

class Classifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=9):
        super().__init__()
        # Block1: 3 -> 16 -> 32
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # → 32×32×16
        )

        # Block2: 32 -> 32 -> 64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # → 64×16×8
        )

        # Block3: 64 -> 64 -> 128 (pool 없음)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # GAP + FC
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # → (B, C, 1, 1)
        x = x.view(x.size(0), -1)             # → (B, C)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    train_dataset = ImageClassificationDataset(
        image_dir='train',
        label_file='Dataset/train_classification.txt'
    )

    test_dataset = ImageClassificationDataset(
        image_dir='test',
        label_file='Dataset/test_classification.txt'
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier(in_channels=3, num_classes=9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter(log_dir='./runs/classification_cv2_acc')

    os.makedirs("classfier_checkpoints", exist_ok=True)
    best_model_path = "classfier_checkpoints/best_model.pth"
    best_acc = 0.0
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} | Test Acc: {accuracy*100:.2f}%")
        writer.add_scalar('Loss/train', avg_loss, epoch + 1)
        writer.add_scalar('Accuracy/test', accuracy, epoch + 1)

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, best_model_path)
            print(f"New best model (acc={best_acc*100:.2f}%) to {best_model_path}")

    writer.close()