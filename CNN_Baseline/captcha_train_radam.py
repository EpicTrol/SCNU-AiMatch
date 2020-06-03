# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import my_dataset
from captcha_VGG16 import *
from captcha_resnet import ResNet18
from captcha_optimizer import RAdam, GradualWarmupScheduler
from captcha_resnet34 import ResNet34

# 断点续训标识
RESUME = True
checkpoint_path = r'./checkpoint/ckpt_res34_2.pth'

# Hyper Parameters
num_epochs = 150
# num_epochs = 1
# batch_size = 100
# batch_size = 64
batch_size = 100
# learning_rate = 5e-4
learning_rate = 1e-5

# Train the Model
train_dataloader = my_dataset.get_train_data_loader(batch_size)

if __name__ == '__main__':
    # cnn = CNN()
    # model = CNN2()
    model = ResNet34()
    model.train()

    model = model.cuda()
    print('init net')
    criterion = nn.MultiLabelSoftMarginLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = RAdam(model.parameters(),
                      lr=learning_rate,
                      betas=(0.9, 0.999),
                      weight_decay=6.5e-4)
    scheduler_after = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=20,
                                                gamma=0.5)
    scheduler = GradualWarmupScheduler(optimizer,
                                       8,
                                       10,
                                       after_scheduler=scheduler_after)
    # 开始训练的epoch
    start_epoch = -1

    # 如果RESUME==True，加载已保存的模型
    if RESUME:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['start_epoch']
        scheduler.load_state_dict(checkpoint['lr_schedule'])
        print('Continue training from epoch {}...'.format(start_epoch))


    for epoch in range(start_epoch+1 ,num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            # 数据移入GPU
            
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()                       # 梯度清零
            predict_labels = model(images)              # 前向传播
            loss = criterion(predict_labels, labels)    # 损失计算
            loss.backward()                             # 反向传播
            optimizer.step()                            # 参数更新

            if (i+1) % 10 == 0:
                print("epoch:", epoch, "step:", i+1, "loss:", loss.item())

            # if (i+1) % 100 == 0:
            #     torch.save(model.state_dict(), "model.pkl")   #current is model.pkl
            #     print("save model")
        print("epoch:", epoch, "finished!", "loss:", loss.item(), "learning_rate:", scheduler.get_lr())

        checkpoint = {
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'start_epoch': epoch,
            'lr_schedule': scheduler.state_dict()
        }

        scheduler.step()
        
        # 如果路径不存在，创建
        if not os.path.isdir("./checkpoint"):
            os.mkdir("./checkpoint")
        torch.save(checkpoint, './checkpoint/ckpt_res34_2.pth')

    # torch.save(model.state_dict(), "model.pkl")               #current is model.pkl
    torch.save(checkpoint, './checkpoint/ckpt_res34_last_2.pth')
    print("save last model")

