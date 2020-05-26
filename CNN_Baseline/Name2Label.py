import os
import pandas as pd
import shutil

if __name__ == '__main__':
    # 训练集路径
    curFilepath = r"D:\IDM\Download\captcha.Pytorch-master_5\captcha.Pytorch-master\dataset\train4"
    # 标签路径
    csvPath = r"D:\IDM\Download\captcha.Pytorch-master_5\captcha.Pytorch-master\dataset\train_label.csv"
    # 转换的目标路径
    newFilePath = r"D:\IDM\Download\captcha.Pytorch-master_5\captcha.Pytorch-master\dataset\train_4"
    
    csv_name = []
    csv_file = pd.read_csv(csvPath)                                 # 按序读标签字符到列表中

    for i in range(20000):

        fileType = '.jpg'
        csv_name.append(csv_file.iloc[i,1])                         # 每次取一个放进列表
        
        # print(csv_name[0],str(i+1))
        older = os.path.join(curFilepath, str(i+1)+fileType)        # 旧文件路径
        newer = os.path.join(curFilepath, csv_name[0]+fileType)     # 新命名文件路径

        nfolder = os.path.join(newFilePath, csv_name[0]+fileType)   # 拷贝文件的路径
        try:
            os.rename(older, newer)                                 # 重命名文件
            shutil.copyfile(newer, nfolder)                         # 拷贝文件到新目录
            csv_name.pop()                                          
        except FileExistsError as e:                                # 重命名失败，已存在相同名字（系统不区分大小写导致的
            print(csv_name[0],str(i+1))                             # 打印命名失败的文件列表，一共201个
            shutil.copyfile(newer, nfolder)
            csv_name.pop()
        except FileNotFoundError as e:
            pass
        continue


