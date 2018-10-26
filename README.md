uint8 cnn(python 实现)
====

    本项目其实就是用python来复现谷歌的这篇论文[Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)<br>
论文虽然简单，谷歌官方也提供了uint8的一整套实现方案，但按照官方文档一步步做下来后，你最后得到的是一个*.tflite文件，也就是说把内部实现细节给封装了，只给你预留了输入和输出接口。。一来不方便我们了解内部实现原理，二来限制了使用场合（似乎只能在Android端结合tensorflow lite来用）。
