快速开始
====
- __步骤一：10分钟环境安装__

    Python3.7 、ImageCaptcha库(pip install captcha)、 Pytorch(参考官网http://pytorch.org)

- __步骤二：训练模型__
    ```bash
    python captcha_train.py
    ```
    使用步骤一生成的验证码图集合用CNN模型（在 catcha_cnn_model 中定义）进行训练，训练完成会生成文件 model.pkl

- __步骤三：测试模型__
    ```bash
    python captcha_test.py
    ```
    可以在控制台，看到模型的准确率（如 95%） ，如果准确率较低，回到步骤一，生成更多的图片集合再次训练

- __步骤四：使用模型做预测__
    ```bash
    python captcha_predict.py
    ```
    可以在控制台，看到预测输出的结果
    
