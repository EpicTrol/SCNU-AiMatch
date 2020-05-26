# -*- coding: UTF-8 -*-
import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from PIL import Image
import one_hot_encoding as ohe
import captcha_setting
import numpy as np
import matplotlib.pyplot as plt

class mydataset(Dataset):

    def __init__(self, folder, transform=None):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.transform = transform

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        image = image.resize((120, 40), Image.BICUBIC)      # 105*35 -> 120*40
        if self.transform is not None:
            image = self.transform(image)
        label = ohe.encode(image_name.split('.')[0]) # 为了方便，在生成图片的时候，图片文件的命名格式 "4个数字或者数字_时间戳.PNG", 4个字母或者即是图片的验证码的值，字母大写,同时对该值做 one-hot 处理
        return image, label

transform = transforms.Compose([
    # transforms.ColorJitter(),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def get_train_data_loader():

    dataset = mydataset(captcha_setting.TRAIN_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=64, shuffle=True)

def get_test_data_loader():
    dataset = mydataset(captcha_setting.TEST_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)

def get_predict_data_loader():
    dataset = mydataset(captcha_setting.PREDICT_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)

if __name__ == "__main__":
    # train_loader = get_train_data_loader()
    # for i, (im, lb) in enumerate(train_loader):
    #     print(i, im.shape, lb.shape)
    #     print(im[0])
    #     print(lb[0])
    #
    #     image = np.array(im[0])
    #     print(ohe.decode(np.array(lb[0])))
    #     plt.imshow(image.transpose(1,2,0))
    #     plt.show()
    #
    #     if i == 0:
    #         break

    test_loader = get_test_data_loader()
    for i, (im, lb) in  enumerate(test_loader):
        print(i, im[i].shape, lb[i].shape)

        if i == 0: break
           
