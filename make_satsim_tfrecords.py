"""
TFRecord Builder

Author: 1st Lt Ian McQuaid
Date: 23 March 2019
"""

import cv2
import os
import json
import astropy.io.fits
import tensorflow as tf
import numpy as np


def draw_bounding_boxes(image, obj_dict_gt, save_path="./test_img.png", nchnls=1, nbits=16):
    if nchnls == 1:
        # replicate 1 channel to 3 channels:
        image = np.squeeze(np.stack((image,) * 3, -1))

    clr_grn = (0, 255, 0)
    if nbits == 16:
        clr_grn = (0, 65535, 0)

    draw_gt_boxes(image, obj_dict_gt, clr_grn)

    # write the image with bounding boxes to file
    png_compression = 0  # lowest compression, highest quality, largest file size
    if nbits == 16:
        cv2.imwrite(save_path, image.astype('uint16'), [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
    else:
        cv2.imwrite(save_path, image.astype('uint8'), [cv2.IMWRITE_PNG_COMPRESSION, png_compression])


def draw_gt_boxes(image, obj_dict_gt, color):
    shape = image.shape

    h = shape[0]
    w = shape[1]

    # obj_dict_gt could be empty: must guard against
    if len(obj_dict_gt) > 0:
        for obj in obj_dict_gt:
            x0 = int(obj['x_min'] * w)
            y0 = int(obj['y_min'] * h)
            x1 = int(obj['x_max'] * w)
            y1 = int(obj['y_max'] * h)

            cv2.rectangle(image, (x0, y0), (x1, y1), color, 1)
            cv2.putText(image,
                        obj['class_name'],
                        (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.30,
                        color, 1, cv2.LINE_AA)

    return image


def read_fits(filepath):
    """Reads simple 1-hdu FITS file into a numpy arrays

    Parameters
    ----------
    filepath : str
        Filepath to read the array from
    """
    a = astropy.io.fits.getdata(filepath)
    a = a.astype(np.uint16)

    return a


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def write_tfrecord(image_list, annotation_list, tfrecord_path):
    # Open up a writer for this TFRecord
    writer = tf.python_io.TFRecordWriter(tfrecord_path)

    # Write each example to file
    count = 0
    num_images = len(image_list)
    for image_path, annotation_path in zip(image_list, annotation_list):
        if (count % 100) == 0:
            print("On image " + str(count) + " of " + str(num_images))

        # Read in the files for this example
        image = read_fits(image_path)
        fp = open(annotation_path, "r")
        annotations = json.load(fp)["data"]
        fp.close()

        sequence = [file.encode() for file in annotations["file"]["sequence"]]
        class_name = [obj["class_name"].encode() for obj in annotations["objects"]]
        class_id = [obj["class_id"] for obj in annotations["objects"]]
        y_min = [obj["y_min"] for obj in annotations["objects"]]
        y_max = [obj["y_max"] for obj in annotations["objects"]]
        x_min = [obj["x_min"] for obj in annotations["objects"]]
        x_max = [obj["x_max"] for obj in annotations["objects"]]
        source = [obj["source"].encode() for obj in annotations["objects"]]
        magnitude = [obj["magnitude"] for obj in annotations["objects"]]

        dir_name = annotations["file"]["dirname"]
        file_name = annotations["file"]["filename"]

        # Remove the extension (to be compatible with what Greg did...)
        file_name = file_name.replace('.fits', '')

        # Put the two together via '_' (again, to be compatible...)
        path_name = dir_name + "_" + file_name

        # Replace the unknown magnitude's with NaN's
        for i in range(len(magnitude)):
            if magnitude[i] is None:
                magnitude[i] = float("NaN")

        # Create the features for this example
        features = {
            "images_raw": _bytes_feature([image.tostring()]),
            "filename": _bytes_feature([path_name.encode()]),
            "height": _int64_feature([annotations["sensor"]["height"]]),
            "width": _int64_feature([annotations["sensor"]["width"]]),
            "classes": _int64_feature(class_id),
            "ymin": _floats_feature(y_min),
            "ymax": _floats_feature(y_max),
            "xmin": _floats_feature(x_min),
            "xmax": _floats_feature(x_max),
        }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=features))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        count += 1


def parse_file_list(file_name):
    satnet_path = os.path.join(os.getcwd(), "SatNet.v.1.0.0.0", "SatNet")
    split_path = os.path.join(satnet_path, "info", "data_split", file_name)
    data_path = os.path.join(satnet_path, "data")

    # Read the split file in
    fp = open(split_path, "r")
    file_contents = fp.read()
    fp.close()

    # Split by line break
    file_list = file_contents.split("\n")

    # Remove the extension (we will add them back in later)
    file_list = [".".join(name.split(".")[:-1]) for name in file_list]

    # Split into the collect name and file name (because string manipulation is fun!)
    collect_list = [name.split("_")[0] for name in file_list]
    file_list = ["_".join(name.split("_")[1:]) for name in file_list]

    image_list = []
    annotation_list = []
    for i in range(len(collect_list)):
        collect_path = collect_list[i]
        file_name = file_list[i]
        if len(file_name) > 0:
            # Build the path to each .fits image
            img_path = os.path.join(data_path, collect_path, "ImageFiles", file_name + ".fits")
            annotation_path = os.path.join(data_path, collect_path, "Annotations", file_name + ".json")
            image_list.append(img_path)
            annotation_list.append(annotation_path)

    # Make a list for images and for annotations
    return image_list, annotation_list


if __name__ == "__main__":
    print("Script starting...")

    # Read in the files that we need in each individual TFRecord
    train_images, train_annotations = parse_file_list("train.txt")
    valid_images, valid_annotations = parse_file_list("valid.txt")
    test_images, test_annotations = parse_file_list("test.txt")

    write_tfrecord(train_images, train_annotations, "train.tfrecords")
    write_tfrecord(valid_images, valid_annotations, "valid.tfrecords")
    write_tfrecord(test_images,  test_annotations, "test.tfrecords")

    print("Script complete. Exiting...")
