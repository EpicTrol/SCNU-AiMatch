# SCNU-AiMatch

类似比赛教程

+ https://www.kaggle.com/shawon10/captcha-recognition
+ https://github.com/dee1024/pytorch-captcha-recognition
  >作者的model是一个五层CNN，经过多次调参，测试集准确率最高到50%。
   模型（五层CNN）可能过于简单（也可能调参不够好），于是将模型换为了34层的残差网络res34-net，达到了测试集上86%的准确率：1W例训练集、1000例测试集、15-30个epoch、64大小batch_size、学习率0.0002，loss最终在0.001左右。
+ https://github.com/ypwhs/baiduyun_deeplearning_competition
  
  >本代码来自群里的示例代码 根据这个作者修改的
  >
  >主要环境：tensorflow keras 所用模块：pillow h5py pillow captcha
  
+ https://github.com/skyduy/CNN_keras/
+ 

## 具体步骤：

### 文件大致结构

将压缩包解压后大致为这样的目录，训练集2w张图片，测试集1w张图片

```
├─test
│  │  1.jpg       		
│  │  ......				
│  │  ......				
│  └─ 10000.jpg
│  
├─train
│  │  1.jpg       		
│  │  ......				
│  │  20000.jpg				
│  └─ train_label.csv
│
├─logs	保存训练的模型文件					
│ └─ xxxx.h5
│
├─predict.py 
|
├─model.py
|
└─README.md 				
```



### 预处理

（1）将整张图片切割为多个小图片，每个小图片包含一个字符

（2）训练模型，识别每个小图片中的字符

（3）将每个小图片的识别出的字符拼接为字符串作为整体识别结果

### 思路

摘自忘了在哪的教程

这其实是一个多标签分类问题，每个验证码图片有4个字符（标签），并且顺序固定；只要将卷积神经网络的最后一层softmax稍加修改就能实现多标签分类。

假设我们的验证码一共有4个字符,每个字符取26个大写字母中的一个；将卷积神经网络的输出层激活函数修改为sigmoid，输出层的[0-25]输出值对应第一个字符的onehot编码，[26-51]输出值对应第二个字符的onehot编码，[52-77]输出值对应第三个字符，[78-103]输出值对于第四个字符，并使用binary_crossentropy作为损失函数。

训练和提交结果
====
| Model        | Learning_rate | Epoch | Optimizer | Train:Test      | Accuracy(Test:Submit) |
| ------------ | ------------- | ----- | --------- | --------------- | --------------------- |
| CNN_baseline | 1e-3          | 30    | Adam      | 9:1(NotRandom)  | None / 0.4018           |
| VGG-16(Mini) | 5e-4          | 30    | Adam      | 9:1             | 81% / 0.8141            |
|              | 5e-4          | 100   | Adam      | 9:1             | 85% / 0.8557            |
| VGG-16       | 5e-4          | 100   | Adam      | 9:1             | 89.33% / 0.9089         |
| ResNet18     | 3e-4          | 100   | Adam      | 9:1             | 93.53% / 0.9379         |
|              | 3e-4+3e-5     | 100   | Adam      | 9:1             | 95.50% / 0.9520         |



更新日志
====
+ 2020/5/26 版本0.1 CNN_Baseline 提交准确率40% 
+ 2020/5/27 版本0.2 VGG-16_mini  提交准确率80%
+ 2020/5/28 版本0.3 VGG-16       提交准确率90%
+ 2020/5/30 版本0.4 ResNet18     提交准确率93%
+ 2020/6/1  版本0.4.1 ResNet18   提交准确率95%
