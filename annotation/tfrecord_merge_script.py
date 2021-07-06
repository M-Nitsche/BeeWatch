import tensorflow as tf
import os

list_of_tfrecord_files = [os.path.join("bees_demo1_split", "bees_demo1.tfrecord"), os.path.join("bees_demo2_split", "bees_demo2.tfrecord")]
dataset = tf.data.TFRecordDataset(list_of_tfrecord_files)

# Save dataset to .tfrecord file
filename = 'bees_demo12.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(dataset)