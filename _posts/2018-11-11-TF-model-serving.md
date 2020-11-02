---
layout: post
title: The complete guide of serving TensorFlow / Keras model in python
date: 2018-11-11 01:09:00
tags: tutorial
---

> This notebook explain how do we serialize the trained TF (keras) models and serve them in python package.

The **tf.train.Saver** class provides methods to save and restore model variables. This require names of the tensors to be known, and we do not discuss here.

## Save TF model

Use SavedModel to save and load your model—variables, the graph, and the graph's metadata. This is a language-neutral, recoverable, hermetic serialization format that enables higher-level systems and tools to produce, consume, and transform TensorFlow models.

### 0. Inspect nodes and tensors
sess.graph.get_operations() gives you a list of operations. For an op, op.name gives you the name and op.values() gives you a list of tensors it produces

### 1. Simple save
The **tf.saved_model.simple_save** function is an easy way to build a tf.saved_model suitable for serving.
```python
tf.saved_model.simple_save(session,
            export_dir,
            inputs={"x": x, "y": y},
            outputs={"z": z})
```
The tensor names of x,y and z defined in the computation graph need to know when loading the model and getting tensors for running model.

### 2. Manually build a SavedModel
The **tf.saved_model.builder.SavedModelBuilder** class provides functionality to save multiple MetaGraphDefs. A MetaGraph is a dataflow graph, plus its associated variables, assets, and signatures. A MetaGraphDef is the protocol buffer representation of a MetaGraph. A signature is the set of inputs to and outputs from a graph.

If assets need to be saved and written or copied to disk, they can be provided when the first MetaGraphDef is added. If multiple MetaGraphDefs are associated with an asset of the same name, only the first version is retained.

Each MetaGraphDef added to the SavedModel must be annotated with user-specified tags. The tags provide a means to identify the specific MetaGraphDef to load and restore, along with the shared set of variables and assets. These tags typically annotate a MetaGraphDef with its functionality (for example, serving or training), and optionally with hardware-specific aspects (for example, GPU).
```python
export_dir = ...
...
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph_and_variables(sess,
                                       [tag_constants.TRAINING],
                                       signature_def_map=foo_signatures,
                                       assets_collection=foo_assets,
                                       strip_default_attrs=True)
...
# Add a second MetaGraphDef for inference.
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph([tag_constants.SERVING], strip_default_attrs=True)
...
builder.save()
```

### 3. Structure of a SavedModel directory
When you save a model in SavedModel format, TensorFlow creates a SavedModel directory consisting of the following subdirectories and files:
```bash
assets/
assets.extra/
variables/
    variables.data-?????-of-?????
    variables.index
saved_model.pb|saved_model.pbtxt
```
where:

* assets is a subfolder containing auxiliary (external) files, such as vocabularies. Assets are copied to the SavedModel location and can be read when loading a specific MetaGraphDef.
* assets.extra is a subfolder where higher-level libraries and users can add their own assets that co-exist with the model, but are not loaded by the graph. This subfolder is not managed by the SavedModel libraries.
* variables is a subfolder that includes output from tf.train.Saver.
* saved_model.pb or saved_model.pbtxt is the SavedModel protocol buffer. It includes the graph definitions as MetaGraphDef protocol buffers.


## Freeze TF model

https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

When serving the model, ususally it's converted to a frozen graph, meaning convert all variables

```python
def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def
freeze_graph('./checkpoints', 'ppscore')
```
After running this function, you should see the printout:
```
INFO:tensorflow:Restoring parameters from ./checkpoints/var
INFO:tensorflow:Froze 2 variables.
Converted 2 variables to const ops.
51 ops in the final graph.
```

## Load TF model and run prediction

### 1. Load saved model

The Python version of the SavedModel tf.saved_model.loader provides load and restore capability for a SavedModel. The load operation requires the following information:

* The session in which to restore the graph definition and variables.
* The tags used to identify the MetaGraphDef to load.
* The location (directory) of the SavedModel.
Upon a load, the subset of variables, assets, and signatures supplied as part of the specific MetaGraphDef will be restored into the supplied session.

```python
loadgraph = tf.Graph()
#with loadgraph.as_default():
with tf.Session(graph=loadgraph) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], save_path_model)
    tf_train_vec1_set = loadgraph.get_tensor_by_name('sent1:0')
    tf_train_vec2_set = loadgraph.get_tensor_by_name('sent2:0')
    ppscore = loadgraph.get_tensor_by_name('ppscore:0')
    scores = sess.run(ppscore, {tf_train_vec1_set: train_vec1_set[:1],
                                tf_train_vec2_set: train_vec2_set[:1]})
    print(scores)
```
where sent1, sen2 and ppscore are tensor names defined in computation graph.


### 2. Load frozen graph

Parsing the frozen graph file and get a graph.
```python
def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph
graph = load_graph(frozen_model_filename)
```

Load the graph into a session, and feed data to gotten tensors for calculation.

```python
# We access the input and output nodes
x1 = graph.get_tensor_by_name('prefix/sent1:0')
x2 = graph.get_tensor_by_name('prefix/sent2:0')
y = graph.get_tensor_by_name('prefix/ppscore:0')

# We launch a Session
with tf.Session(graph=graph) as sess:
    # Note: we don't nee to initialize/restore anything
    # There is no Variables in this graph, only hardcoded constants
    y_out = sess.run(y, feed_dict={
        x1: train_vec1_set[:2],
        x2: train_vec2_set[:2]# < 45
    })
    # I taught a neural net to recognise when a sum of numbers is bigger than 45
    # it should return False in this case
    print(y_out) # [[ False ]] Yay, it works!
```

Here is a tip for look into the model. After we load the graph, we could also use the following codes to see what operations are available.

```python
# We can verify that we can access the list of operations in the graph
for op in graph.get_operations():
    print(op.name)
    # prefix/Placeholder/inputs_placeholder
    # ...
    # prefix/Accuracy/predictions
```

## Convert keras Model to TF pb file

### 1. Save pb file


Keras model h5 file also can be converted to a tensorflow frozen graph pb file.
```python
import tensorflow as tf
import numpy as np

import os
import os.path as osp
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras.models import load_model
from keras import backend as K

# Create function to convert saved keras model to tensorflow graph
def convert_to_pb(weight_file,input_fld='',output_fld=''):
    # weight_file is a .h5 keras model file
    output_node_names_of_input_network = ["pred0"]
    output_node_names_of_final_network = 'output_node'

    # change filename to a .pb tensorflow file
    output_graph_name = weight_file[:-2]+'pb'
    weight_file_path = osp.join(input_fld, weight_file)

    net_model = load_model(weight_file_path)

    num_output = len(output_node_names_of_input_network)
    pred = [None]*num_output
    pred_node_names = [None]*num_output

    for i in range(num_output):
        pred_node_names[i] = output_node_names_of_final_network+str(i)
        pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])

    sess = K.get_session()

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
    print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))

    return output_fld+output_graph_name

tf_model_path = convert_to_pb('prod_model.h5','./models/','./models/')
```
After running the convert_to_pb function, you should see the printout:
```
INFO:tensorflow:Froze 4 variables.
Converted 4 variables to const ops.
saved the constant graph (ready for inference) at:  ./models/prod_model.pb
```

### 2. Load and use the model

```python
def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )

    input_name = graph.get_operations()[0].name+':0'
    output_name = graph.get_operations()[-1].name+':0'

    return graph, input_name, output_name

def predict(model_path, input_data):
    # load tf graph
    tf_model,tf_input,tf_output = load_graph(model_path)

    # Create tensors for model input and output
    x = tf_model.get_tensor_by_name(tf_input)
    y = tf_model.get_tensor_by_name(tf_output)

    # Number of model outputs
    num_outputs = y.shape.as_list()[0]
    predictions = np.zeros((input_data.shape[0],num_outputs))
    for i in range(input_data.shape[0]):
        with tf.Session(graph=tf_model) as sess:
            y_out = sess.run(y, feed_dict={x: input_data[i:i+1]})
            predictions[i] = y_out

    return predictions
```

Run the model for prediction:
```python
predict('./models/prod_model.pb',x_val)
```

## Convert and use tflite

### 1. Convert pd file and save tflite file

```python
def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )

    input_name = graph.get_operations()[0].name+':0'
    output_name = graph.get_operations()[-1].name+':0'

    return graph, input_name, output_name
```

```python
model_path = "./models/prod_model.pb"

# load tf graph
tf_model,tf_input,tf_output = load_graph(model_path)
print(tf_input, tf_output)
```

```python
# Create tensors for model input and output
x = tf_model.get_tensor_by_name(tf_input)
y = tf_model.get_tensor_by_name(tf_output)

tflite_filename ="pbsessconverted_model.tflite"
with tf.Session(graph = tf_model) as sess:
    converter = tf.contrib.lite.TocoConverter.from_session(sess, [x], [y])
    tflite_model = converter.convert()
    open(tflite_filename, "wb").write(tflite_model)
```

We can verify that we can access the list of operations in the graph:
```python
for op in tf_model.get_operations():
    print(tensor_name(op))
```


### 2. Load and predict with tflite file

```python
# Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_path=tflite_filename)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
#input_shape = input_details[0]['shape']
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], x_val.astype(np.float32))

# Get output result
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
```


```python

```
