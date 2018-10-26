uint8 cnn(python 实现)
====

本项目其实就是用python来复现谷歌的这篇论文[Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)（Mobilenet_V1和Mobilenet_V2）<br>
论文虽然简单，谷歌官方也提供了uint8的一整套实现方案，但按照官方文档一步步做下来后，`你最后得到的是一个*.tflite文件，也就是说官方把内部实现细节给封装了，只给你预留了输入和输出接口`。一来不方便我们了解内部实现原理，二来限制了使用场合（似乎只能在Android端结合tensorflow lite来用）。<br>
而本项目则是从论文的原理出发，用python来复现论文提出的uint8的一整套方案。（实在不想重写卷积，就偷偷调用了tf.nn.conv2d的高级API）。目前复现的只有Mobilenet_V1和Mobilenet_V2，采用的是Oxford_102鲜花数据集。我在本地测试了一下我的python复现版本和tflite版本，在测试集上的正确率差距不超0.5%。

#用法


