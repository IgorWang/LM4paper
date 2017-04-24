# Tensorflow RNN

## Data feed

numpy arrays or TFRecord ?

I have different datasets in this project. For convenience，I have to
set a unified way to feed data to model. Numpy is memory consumption
especially in big datasets. So I first use TFRecord as my choice.


Basice processing:

> - Convert your data into tf.SequenceExample format
> - Write one or more TFRecord files with the serialized data
> - Use tf.TFRecordReader to read examples from the file
> - Parse each example using tf.parse_single_sequence_example

## Dynamic RNN



## Reference

[Data Input](https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/)

