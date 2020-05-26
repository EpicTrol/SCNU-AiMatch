快速开始
====
- __步骤一：10分钟环境安装__

    Python2.7+ 、ImageCaptcha库(pip install captcha)、 Pytorch(参考官网http://pytorch.org)


- __步骤二：生成验证码__
    ```bash
    python captcha_gen.py
    ```
    执行以上命令，会在目录 dataset/train/ 下生成多张验证码图片，图片已经标注好，数量可以是 1w、5w、10w，通过 captcha-gen.py 内的 count 参数设定
    
- __步骤三：训练模型__
    ```bash
    python captcha_train.py
    ```
    使用步骤一生成的验证码图集合用CNN模型（在 catcha_cnn_model 中定义）进行训练，训练完成会生成文件 model.pkl

- __步骤四：测试模型__
    ```bash
    python captcha_test.py
    ```
    可以在控制台，看到模型的准确率（如 95%） ，如果准确率较低，回到步骤一，生成更多的图片集合再次训练

- __步骤五：使用模型做预测__
    ```bash
    python captcha_predict.py
    ```
    可以在控制台，看到预测输出的结果
    
