---
layout: post
title: A brief overview of the TernsorFlow family
date: 2019-05-31 01:09:00
tags: deep learning frameworks
---

>TensorFlow, which was first relaesed in 2015, is cureerly the largest deep learning framework exists so far. As the core TensorFlow grows, Google has also developed a series of library / frameworks around the core TensorFlow to build up a complete eco-system for deep learning related works. 

Here we briefly go through the current existing libraries / tools in the TensorFlow family so that we get a clear picture of what are available and what should they be properly used for.

In this post, I will try to maintain a comprehensive list of the members in TensorFlow family. This list is orgnized accodring to the order of a machine learning pipeline.

---
## Prep
---

### TensorFlow Datasets 
```python
import tensorflow_datasets as tfds
```
TensorFlow Datasets is a collection of datasets ready to use with TensorFlow. 

* [Official website](https://www.tensorflow.org/datasets)   
* [Introducing TensorFlow Datasets](https://medium.com/tensorflow/introducing-tensorflow-datasets-c7f01f7e19f3)  
* [A collection of datasets ready to use with TensorFlow.](https://www.tensorflow.org/datasets)

---

## Core TensorFlow
---

### TensorFlow Data
```python
import tensorflow as tf
tf.data
```
This is the module for put up your input pipeline.

[Guide of tf.data](https://www.tensorflow.org/guide/datasets)

--- 

### Ragged Tensors
```python
import tensorflow as tf
tf.ragged
```
Your data comes in many shapes; your tensors should too. Ragged tensors are the TensorFlow equivalent of nested variable-length lists. They make it easy to store and process data with non-uniform shapes, including:

* Variable-length features, such as the set of actors in a movie.
* Batches of variable-length sequential inputs, such as sentences or video clips.
* Hierarchical inputs, such as text documents that are subdivided into sections, paragraphs, sentences, and words.
* Individual fields in structured inputs, such as protocol buffers.

[Guide of Ragged Tensors](https://www.tensorflow.org/guide/ragged_tensors)

---

### TensorFlow Keras
```python
import tensorflow as tf
tf.keras
```
Keras is a high-level API to build and train deep learning models. It's used for fast prototyping, advanced research, and production.

[Guide of tf.keras](https://www.tensorflow.org/guide/keras)

---

### TensorFlow Estimator
```python
import tensorflow as tf
tf.estimator
```

Estimators, which should be used for machine learning production development, encapsulate the following actions:  

* training
* evaluation
* prediction
* export for serving

[Guide of Estimators](https://www.tensorflow.org/guide/estimators)

---

### TensorFlow Feature Columns

feature columns—a data structure describing the features that an Estimator requires for training and inference. As you'll see, feature columns are very rich, enabling you to represent a diverse range of data.

[Introducing TensorFlow Feature Columns](https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html)

--- 

### TensorFlow Lite
```python
import tensorflow as tf
tf.lite
```
	 
TensorFlow Lite is an open source deep learning framework for on-device inference.

[TensorFlow Lite guide](https://www.tensorflow.org/lite/guide)

--- 

### TensorFlow Model Optimization Toolkit
```python
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
```
This is a suite of techniques that developers, both novice and advanced, can use to optimize machine learning models for deployment and execution.

[Introducing the Model Optimization Toolkit for TensorFlow](https://medium.com/tensorflow/introducing-the-model-optimization-toolkit-for-tensorflow-254aca1ba0a3)  
[Guide of model optimization](https://www.tensorflow.org/lite/performance/model_optimization)

---

### TensorFlow Debugger 
```python
from tensorflow.python import debug as tf_debug
```

tfdbg is a specialized debugger for TensorFlow. It lets you view the internal structure and states of running TensorFlow graphs during training and inference, which is difficult to debug with general-purpose debuggers such as Python's pdb due to TensorFlow's computation-graph paradigm.

[Guide of TensorFlow Debugger](https://www.tensorflow.org/guide/debugger)  
[Debug TensorFlow Models with tfdbg](https://developers.googleblog.com/2017/02/debug-tensorflow-models-with-tfdbg.html)

---

## Suites
---

### TensorFlow Probability
```python
import tensorflow_probability as tfp
```

A probabilistic programming toolbox for machine learning researchers and practitioners to quickly and reliably build sophisticated models that leverage state-of-the-art hardware.


[Official website](https://www.tensorflow.org/probability)  
[Introducing TensorFlow Probability](https://medium.com/tensorflow/introducing-tensorflow-probability-dca4c304e245)  
[Guide of TensorFlow Probability](https://www.tensorflow.org/probability/overview)  

---

### TF-Agents
```python
import tf_agents
```
The library for reinforcement learning with TensorFlow.

[TF-Agents github](https://github.com/tensorflow/agents)

---

### TensorFlow Federated (TFF)

TensorFlow Federated (TFF) is an open source framework for experimenting with machine learning and other computations on decentralized data. It implements an approach called Federated Learning (FL), which enables many participating clients to train shared ML models, while keeping their data locally. 

*Note: Current relased version only supports local simulation.*

[Introducing TensorFlow Federated](https://medium.com/tensorflow/introducing-tensorflow-federated-a4147aa20041)

---

### TensorFlow Privacy

an open source library that makes it easier not only for developers to train machine-learning models with privacy, but also for researchers to advance the state of the art in machine learning with strong privacy guarantees.

[Introducing TensorFlow Privacy: Learning with Differential Privacy for Training Data](https://medium.com/tensorflow/introducing-tensorflow-privacy-learning-with-differential-privacy-for-training-data-b143c5e801b6)

---

### TensorFlow Ranking

TensorFlow Ranking is a library for Learning-to-Rank (LTR) techniques on the TensorFlow platform. It contains the following components:  

* Commonly used loss functions including pointwise, pairwise, and listwise losses.
* Commonly used ranking metrics like Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG).
* Multi-item (also known as groupwise) scoring functions.
* LambdaLoss implementation for direct ranking metric optimization.
* Unbiased Learning-to-Rank from biased feedback data.
  
We envision that this library will provide a convenient open platform for hosting and advancing state-of-the-art ranking models based on deep learning techniques, and thus facilitate both academic research as well as industrial applications.

[TF Ranking github](https://github.com/tensorflow/ranking)

---

### TFX
```python
from tfx.utils.dsl_utils import tfrecord_input
from tfx.components import ...
```
TensorFlow Extended is an end-to-end platform for preparing data, training, validating, and deploying models in large production environments.

When you’re ready to go beyond training a single model, or ready to put your amazing model to work and move it to production, TFX is there to help you build a complete ML pipeline.

A TFX pipeline is a sequence of components that implement an ML pipeline which is specifically designed for scalable, high-performance machine learning tasks. That includes modeling, training, serving inference, and managing deployments to online, native mobile, and JavaScript targets. To learn more, read our TFX User Guide.

A TFX pipeline typically includes the following components:  

* **ExampleGen** is the initial input component of a pipeline that ingests and optionally splits the input dataset.
* **StatisticsGen** calculates statistics for the dataset.
* **SchemaGen** examines the statistics and creates a data schema.
* **ExampleValidator** looks for anomalies and missing values in the dataset.
* **Transform** performs feature engineering on the dataset.
* **Trainer** trains the model.
* **Evaluator** performs deep analysis of the training results.
* **ModelValidator** helps you validate your exported models, ensuring that they are "good enough" to be pushed to production.
* **Pusher** deploys the model on a serving infrastructure.

This diagram illustrates the flow of data between these components:
![Reinforcement Learning Algorithms]({{'/images/TFX.jpg'|relative_url}})

[Official website](https://www.tensorflow.org/tfx)  
[Guide of TFX](https://www.tensorflow.org/tfx/guide)

---

### Mesh-TensorFlow
```python
import mesh_tensorflow.auto_mtf
```
Mesh TensorFlow (mtf) is a language for distributed deep learning, capable of specifying a broad class of distributed tensor computations. The purpose of Mesh TensorFlow is to formalize and implement distribution strategies for your computation graph over your hardware/processors. For example: "Split the batch over rows of processors and split the units in the hidden layer across columns of processors." Mesh TensorFlow is implemented as a layer over TensorFlow.

[Github](https://github.com/tensorflow/mesh)

---

## Utils

---

### TensorBoard
 You can use TensorBoard to visualize your TensorFlow graph, plot quantitative metrics about the execution of your graph, and show additional data like images that pass through it. 

[Official website](https://www.tensorflow.org/guide/summaries_and_tensorboard)

--- 

### Model optimization

The TensorFlow Model Optimization Toolkit is a suite of tools for optimizing ML models for deployment and execution. Among many uses, the toolkit supports techniques used to:  

* Reduce latency and inference cost for cloud and edge devices (e.g. mobile, IoT).
* Deploy models to edge devices with restrictions on processing, memory, power-consumption, network usage, and model storage space.
* Enable execution on and optimize for existing hardware or new special purpose accelerators.

[Official website](https://www.tensorflow.org/model_optimization)  
[Guide of TensorFlow model optimization](https://www.tensorflow.org/model_optimization/guide)

---

### TensorFlow Hub

TensorFlow Hub is a library for the publication, discovery, and consumption of reusable parts of machine learning models. A module is a self-contained piece of a TensorFlow graph, along with its weights and assets, that can be reused across different tasks in a process known as transfer learning. Transfer learning can:

* Train a model with a smaller dataset,
* Improve generalization, and
* Speed up training.

[Introducing TensorFlow Hub](https://medium.com/tensorflow/introducing-tensorflow-hub-a-library-for-reusable-machine-learning-modules-in-tensorflow-cdee41fa18f9)  
[Guide of TensorFlow Hub](https://www.tensorflow.org/hub/overview)

---

### TesnsorFlow Serving

TensorFlow Serving is a flexible, high-performance serving system for machine learning models, designed for production environments. TensorFlow Serving makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. TensorFlow Serving provides out-of-the-box integration with TensorFlow models, but can be easily extended to serve other types of models and data.

[Guide of TF Serving](https://www.tensorflow.org/tfx/guide/serving)
[Github of TF Serving](https://github.com/tensorflow/serving.git)

---

## TensorFLow in other Language

---

### TensorFLow.js

TensorFlow.js is a library for developing and training ML models in JavaScript, and deploying in browser or on Node.js

[Official website](https://www.tensorflow.org/js)

---

### TensorFLow.jl

TensorFlow.jl is a library for developing and training ML models in Julia.

[Official website](https://www.tensorflow.org/jl)

---

### Swift for TensorFlow

Swift for TensorFlow is a next generation platform for deep learning and differentiable programming.
By integrating directly with a general purpose programming language, Swift for TensorFlow enables more powerful algorithms to be expressed like never before.

[Official website](https://www.tensorflow.org/swift)


 