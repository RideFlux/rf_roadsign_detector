import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class ImageClassificationDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []
        with open(label_file, 'r') as f:
            for line in f:
                name, label = line.strip().split()
                self.samples.append((name, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(filename)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        return image, label

class Classifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


train_dataset = ImageClassificationDataset(
    image_dir='train',
    label_file='Dataset/label.txt'
)

test_dataset = ImageClassificationDataset(
    image_dir='test',
    label_file='Dataset/label.txt'
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Classifier(in_channels=1, num_classes=9).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
writer = SummaryWriter(log_dir='./runs/classification_cv2_acc')

os.makedirs("checkpoints", exist_ok=True)
best_model_path = "classfier_checkpoints/best_model.pth"
best_acc = 0.0
num_epochs = 20

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