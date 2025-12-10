import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from roadsign_classifier import ImageClassificationDataset, Classifier

train_dataset = ImageClassificationDataset(
    image_dir='train',
    label_file='Dataset/train_classification.txt',
    return_file_path=True
)

test_dataset = ImageClassificationDataset(
    image_dir='test',
    label_file='Dataset/test_classification.txt',
    return_file_path=True
)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Classifier(in_channels=3, num_classes=9).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
writer = SummaryWriter(log_dir='./runs/classification_cv2_acc')

checkpoint = torch.load('classfier_checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

for images, labels, file_paths in test_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)

    images = images.to('cpu').detach().numpy() * 255
    images = images.astype(np.uint8)
    # images = images[0]
    labels = labels.to('cpu').detach().numpy()
    outputs = outputs.to('cpu').detach().numpy()
    # print(images.shape)
    # print(labels.shape)
    # print(outputs.shape)
    # print(file_paths)

    for img, label, output, file_path in zip(images, labels, outputs, file_paths):
        output_class = np.argmax(output)
        if 'sh' not in file_path:
            continue
        if label != 0 and label != 1 and output_class != 0 and output_class != 1:
            continue

        saving_img = img.transpose(1, 2, 0)
        
        cv2.imwrite('test.png', saving_img)
        print('file name :', file_path)
        
        print(label, np.argmax(output), output)
        # print(img.shape)

        a = input()
        if a == 'c':
            exit()