# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
#from visdom import Visdom # pip install Visdom
import captcha_setting
import my_dataset
from captcha_cnn_model import CNN
import csv
import one_hot_encoding as oht
import os
from my_dataset import *

# 生成csv文件的路径
csv_file = 'submission.csv'
# predict文件的路径
pathDir = os.listdir('dataset/predict/')

def main():
    f = open(csv_file, "w", newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["ID","label"])

    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pkl'))
    print("load cnn net.")

    # predict_dataloader = my_dataset.get_predict_data_loader()

    #vis = Visdom()
    # for i, (images, labels) in enumerate(predict_dataloader):
    #     image = images
    #     print(image.shape)
    #     # vimage = Variable(image)
    #     predict_label = cnn(image)
    #
    #     c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    #     c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    #     c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    #     c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    #
    #     label = '%s%s%s%s' % (c0, c1, c2, c3)
    #     ID = str(np.array(labels)[0]) + ".jpg"
    #     csv_writer.writerow([ID,label])
    #     print(ID, label)
    #     # print(np.array(labels)[0], c)
    #     #vis.images(image, opts=dict(caption=c))

    # 按照文件名（除去格式部分）排序
    pathDir.sort(key=lambda x: int(x[:-4]))

    for allDir in pathDir:
        file = os.path.join('%s%s' % (pathDir, allDir))
        fopen = Image.open(file)

        image = fopen.resize((120, 40), Image.BICUBIC)
        image = transform(image)
        image = torch.unsqueeze(image, dim=0)

        predict_label = cnn(image)

        c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
            predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
        label = '%s%s%s%s' % (c0, c1, c2, c3)

        ID = str(allDir)
        csv_writer.writerow([ID, label])
        print(ID, label)

    f.close()

if __name__ == '__main__':
    main()


