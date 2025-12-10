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
    label_file='Dataset/train_classification.txt'
)

test_dataset = ImageClassificationDataset(
    image_dir='test',
    label_file='Dataset/test_classification.txt'
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

tp = np.zeros([9], dtype=np.int32)
fp = np.zeros([9], dtype=np.int32)
fn = np.zeros([9], dtype=np.int32)

confusion_matrix = np.zeros([9,9], dtype=np.int32)

for images, labels in test_loader:
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

    for img, label, output in zip(images, labels, outputs):
        # img = img[0]
        # cv2.imwrite('test.png', img)
        # print(label, np.argmax(output), output)
        # print(img.shape)

        model_class = np.argmax(output)
        confusion_matrix[label, model_class] += 1
        if label == model_class:
            tp[label] += 1
        else:
            fn[label] += 1
            fp[model_class] += 1

        # a = input()
        # if a == 'c':
        #     exit()
print('tp :', tp)
print('fp :', fp)
print('fn :', fn)
print(confusion_matrix)
# print(correct / (correct+wrong))