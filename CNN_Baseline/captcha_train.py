# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import my_dataset
from captcha_cnn_model import CNN

# Hyper Parameters
num_epochs = 30
# num_epochs = 1
# batch_size = 100
batch_size = 8
learning_rate = 0.0001

if __name__ == '__main__':
    cnn = CNN()
    cnn.train()
    # 模型移入GPU
    if torch.cuda.is_available():
        cnn = cnn.cuda()
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    train_dataloader = my_dataset.get_train_data_loader()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            # images = Variable(images)
            # labels = Variable(labels.float())
            # 数据移入GPU
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            predict_labels = cnn(images)
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print("epoch:", epoch, "step:", i+1, "loss:", loss.item())
            if (i+1) % 100 == 0:
                torch.save(cnn.state_dict(), "model.pkl")   #current is model.pkl
                print("save model")
        print("epoch:", epoch, "step:", i, "loss:", loss.item())
    torch.save(cnn.state_dict(), "model.pkl")               #current is model.pkl
    print("save last model")

    
