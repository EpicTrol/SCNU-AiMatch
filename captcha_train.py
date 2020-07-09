# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import my_dataset
from captcha_cnn_model import CNN
from captcha_VGG16 import *

# 断点续训标识
RESUME = True
checkpoint_path = r'./checkpoint/ckpt_model.pth'

# Hyper Parameters
num_epochs = 100
# num_epochs = 1
# batch_size = 100
batch_size = 32
learning_rate = 5e-4

# Train the Model
train_dataloader = my_dataset.get_train_data_loader(batch_size)

if __name__ == '__main__':
    # cnn = CNN()
    model = CNN2()
    model.train()

    # 模型移入GPU
    if torch.cuda.is_available():
        model = model.cuda()
    print('init net')
    # print(model)
    criterion = nn.MultiLabelSoftMarginLoss()#.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练的epoch
    start_epoch = -1

    # 如果RESUME==True，加载已保存的模型
    if RESUME:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['start_epoch']
        print('Continue training from epoch {}...'.format(start_epoch))


    for epoch in range(start_epoch+1 ,num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            # 数据移入GPU
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            predict_labels = model(images)
            # print(predict_labels.type)
            # print(labels.type)
            loss = criterion(predict_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print("epoch:", epoch, "step:", i+1, "loss:", loss.item())

            # if (i+1) % 100 == 0:
            #     torch.save(model.state_dict(), "model.pkl")   #current is model.pkl
            #     print("save model")
        print("epoch:", epoch, "finished!", "loss:", loss.item())

        checkpoint = {
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'start_epoch': epoch
        }
        # 如果路径不存在，创建
        if not os.path.isdir("./checkpoint"):
            os.mkdir("./checkpoint")
        torch.save(checkpoint, './checkpoint/ckpt_model.pth')

    # torch.save(model.state_dict(), "model.pkl")               #current is model.pkl
    torch.save(checkpoint, './checkpoint/ckpt_last.pth')
    print("save last model")

