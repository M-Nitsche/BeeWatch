"""
This script converts a VIA project into tfrecords.
VIA project:
    Classes are noted in the region_attributes as: Bee: "1"
    The one stands for the track ID of the bee

arguments:
    -v, --via_dir       Path to the folder where the VIA project is stored
    --l, --labels_path  Path to the labels (.pbtxt) file.
    -o, --output_path   Path of output TFRecord file
    -i, --image_dir     Path to the image folder
    -c, --csv_path      Output path of the CSV file

TFRecords on negative images https://stackoverflow.com/questions/47351307/tensorflow-object-dectection-api-how-to-create-tfrecords-with-images-not-conta
"""

import os
import glob
import pandas as pd
import io
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple
import json
import math

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow XML-to-TFRecord converter")
parser.add_argument("-v",
                    "--via_dir",
                    help="Path to the folder where the input .xml files are stored.",
                    type=str,
                    default="F:/AIinCV/Daten_Oli/6/GP020018_CUT6_crop_f20.json")
parser.add_argument("-l",
                    "--labels_path",
                    help="Path to the labels (.pbtxt) file.", type=str,
                    default="F:/AIinCV/Daten_Oli/label_map.pbtxt")
parser.add_argument("-o",
                    "--output_path",
                    help="Path of output TFRecord (.record) file.", type=str,
                    default="F:/AIinCV/Daten_Oli/6/tfrecords/6_train.record")
parser.add_argument("-i",
                    "--image_dir",
                    help="Path to the folder where the input image files are stored. "
                         "Defaults to the same directory as via_dir.",
                    type=str, default="F:/AIinCV/Daten_Oli/6/")
parser.add_argument("-c",
                    "--csv_path",
                    help="Path of output .csv file. If none provided, then no file will be "
                         "written.",
                    type=str, default="F:/AIinCV/Daten_Oli/6/tfrecords/6_train.csv")
parser.add_argument("-iw",
                    "--image_width",
                    help="Width of the images",
                    type=int, default=850)
parser.add_argument("-ih",
                    "--image_height",
                    help="Height of the images",
                    type=int, default=500)


args = parser.parse_args()

if args.image_dir is None:
    args.image_dir = args.via_dir

label_map = label_map_util.load_labelmap(args.labels_path)
label_map_dict = label_map_util.get_label_map_dict(label_map)

def via_to_csv(path, image_width, image_height):
    """Iterates through all the via project and combines
    it in a single Pandas dataframe.

    Parameters:
    ----------
    path : str
        The path containing the via project JSON file
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """
    with open(path) as via_file:
        via_json = json.load(via_file)
    bb_list = []
    for anno in via_json["_via_img_metadata"].values():
        if not anno["regions"]:
            # image without any bounding box
            # list is empty
            # https://stackoverflow.com/questions/47351307/tensorflow-object-dectection-api-how-to-create-tfrecords-with-images-not-conta
            value = (
                anno["filename"],
                image_width,
                image_height,
                None,
                None,
                None,
                None,
                None
            )
            bb_list.append(value)
        else:
            for reg in anno["regions"]:
                # with bounding box
                value = (
                    anno["filename"],
                    image_width,
                    image_height,
                    list(reg["region_attributes"].keys())[0],
                    int((reg["shape_attributes"])["x"]),
                    int((reg["shape_attributes"])["y"]),
                    (int((reg["shape_attributes"])["x"]) + int((reg["shape_attributes"])["width"])),
                    (int((reg["shape_attributes"])["y"]) + int((reg["shape_attributes"])["height"]))
                )
                bb_list.append(value)

    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    json_df = pd.DataFrame(bb_list, columns=column_name)
    return json_df


def class_text_to_int(row_label):
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = Image.open(encoded_png_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    image_with_bb = True

    for index, row in group.object.iterrows():
        if math.isnan(row['xmin']):
            # image without bee
            image_with_bb = False
            continue
        else:
            image_with_bb = True
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    if not image_with_bb:
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_png),
            'image/format': dataset_util.bytes_feature(image_format)
        }))
    else:
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_png),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(args.output_path)
    path = os.path.join(args.image_dir)
    examples = via_to_csv(args.via_dir, args.image_width, args.image_height)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file: {}'.format(args.output_path))
    if args.csv_path is not None:
        examples.to_csv(args.csv_path, index=None)
        print('Successfully created the CSV file: {}'.format(args.csv_path))


if __name__ == '__main__':
    tf.app.run()





