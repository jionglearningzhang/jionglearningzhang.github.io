---
layout: post
title: About edge AI with TensorFLow!
date: 2019-05-27 23:00:00
tags: review
---

Edge computing is drawing more and more attentions. It is heavily applied in many fields such as intelligent apps on cell phones and in IoTs. Here at Mercedes-Benz R&D, we are also working on advanced development of edge AI frameworks on our Mercedes cars. In this post, I will give an overview of edge AI and how to develop edge AI with TensorFlow.

# 1. High efficient models

In general, high efficient network signifies small number of parameters, small amount of computation, high spped. While they are related to each other, they are indenpent goals to consider.

## Metrics to look at

### Multiply-Accumulates (MACs) 
Multiply-Accumulates (MACs) which measures the number of fused Multiplication and Addition operations. MACs, also sometimes known as MADDs - the number of multiply-accumulates needed to compute an inference on a single image is a common metric to measure the efficiency of the model.



###FLOPs
Floating point operations per second (FLOPS, flops or flop/s) is a measure of computer performance, useful in fields of scientific computations that require floating-point calculations. For such cases it is a more accurate measure than measuring instructions per second.

### memory access cost (MAC)


### mAP

(mean average precision) is the average of AP. AP (Average precision) is a popular metric in measuring the accuracy of object detectors like Faster R-CNN, SSD, etc. Average precision computes the average precision value for recall value over 0 to 1. 

## Popular high efficient networks

### [Xception](https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py)

We present an interpretation of Inception modules in convolutional neural networks, [Xception](https://arxiv.org/abs/1610.02357), as being an intermediate step in-between regular convolution and the depthwise separable convolution operation (a depthwise convolution followed by a pointwise convolution). 



### [CondenseNet](https://github.com/ShichenLiu/CondenseNet)

[CondenseNet](https://arxiv.org/abs/1711.09224) is a novel, computationally efficient convolutional network architecture. It combines dense connectivity between layers with a mechanism to remove unused connections. The dense connectivity facilitates feature re-use in the network, whereas learned group convolutions remove connections between layers for which this feature re-use is superfluous. At test time, our model can be implemented using standard grouped convolutions —- allowing for efficient computation in practice. Our experiments demonstrate that CondenseNets are much more efficient than other compact convolutional networks such as MobileNets and ShuffleNets.

### [MobileNet V1](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)

[MobileNets](https://arxiv.org/abs/1704.04861) are small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. They can be built upon for classification, detection, embeddings and segmentation similar to how other popular large scale models, such as Inception, are used. MobileNets can be run efficiently on mobile devices with TensorFlow Mobile. 

### [MobileNet V2](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)

[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)



### [ShuffleNet V1](https://github.com/MG2033/ShuffleNet)
According to the authors, [ShuffleNet](https://arxiv.org/abs/1707.01083) is a computationally efficient CNN architecture designed specifically for mobile devices with very limited computing power. It outperforms Google MobileNet by small error percentage at much lower FLOPs.

### [ShuffleNet V2](https://github.com/TropComplique/shufflenet-v2-tensorflow)
[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
](https://arxiv.org/abs/1807.11164)


## General thoughts of high efficient network design

0. Only use what is neccesary to the problem (model depth & complexity).
1. depthwise separable convolutions.
2. use low rank/small demension filter.
3. pointwise group convolution.
4. Avoid too many braches or groups.
5. Reduce the number of element wise operations.


# 2. TensorFlow Lite for edge AI

  TensorFlow Lite is an open source deep learning framework for on-device inference. What worth mentioning is that on their roadmap, on-device training will also be supported.
  
 
### TensorFlow Lite converter

Convert a TensorFlow model from proto buffer into a compressed flat buffer, which can be load directly on edge devices.
 
```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```


### TensorFlow Lite's Model Optimization Toolkit

By reducing the precision of values and operations within a model, quantization can reduce both the size of model and the time required for inference. Quantize by converting 32-bit floats to more efficient 8-bit integers

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_quantized_model)
```