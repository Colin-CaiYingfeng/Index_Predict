# Index_Predict
Predict 00001.SH index by Back Propagation Neural Network &amp; Wavelet Neural Network.
### 基于小波BP神经网络的上证指数预测模型
### Wavelet-BP Neural Network

#### Unfinished & Still working on it
#### 暂未完成

##### 模型的灵感来源于中邮证券2018一篇名为《金融时间序列的小波去噪》的研报。
##### 基于BP神经网络对股票市场指数进行预测。

#### feature：
1. 计划采用Morlet小波母函数代替隐藏层的sigmoid作为激活函数
2. MSE作为loss函数
3. 使用df.shift构造监督学习数据集
4. 上传了传统BPnn的demo
5. 上传了SGD随机梯度下降法的demo
