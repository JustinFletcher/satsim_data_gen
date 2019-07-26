"""
TFRecord Builder

Author: 1st Lt Ian McQuaid
Date: 23 March 2019
"""

import os
import json
import shutil

import argparse

import astropy.io.fits
import tensorflow as tf
import numpy as np


from itertools import islice, zip_longest


def pad_arrays_to_same_len(list_of_lists):
    # Get the max array len
    max_len = 0
    for curr_list in list_of_lists:
        max_len = max(max_len, len(curr_list))

    # Now pad each one
    new_list_of_lists = []
    for curr_list in list_of_lists:
        padded_list = np.pad(curr_list,
                             (0, max_len - len(curr_list)),
                             mode='constant',
                             constant_values=0)
        new_list_of_lists.append(padded_list)

    # Done, so return
    return new_list_of_lists


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


def build_satnet_tf_example(example):

    (image_path, annotation_path) = example

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

    return(example)


def build_satnet_multiframe_tf_example(example_sequence):
    sequence_images = []
    sequence_filenames = []
    sequence_classes = []
    sequence_ymin = []
    sequence_ymax = []
    sequence_xmin = []
    sequence_xmax = []
    for example in example_sequence:
        (image_path, annotation_path) = example

        # Read in the files for this example
        image = read_fits(image_path)

        fp = open(annotation_path, "r")
        annotations = json.load(fp)["data"]
        fp.close()

        class_id = [obj["class_id"] for obj in annotations["objects"]]
        y_min = [obj["y_min"] for obj in annotations["objects"]]
        y_max = [obj["y_max"] for obj in annotations["objects"]]
        x_min = [obj["x_min"] for obj in annotations["objects"]]
        x_max = [obj["x_max"] for obj in annotations["objects"]]

        dir_name = annotations["file"]["dirname"]
        file_name = annotations["file"]["filename"]

        new_class_id = []
        for id in class_id:
            if id is 0 or id is 1:
                new_class_id.append(id)
            elif id > 1:
                # Logic to handle stupid bug introduced in last set of fixes
                new_class_id.append(1)
            else:
                print("Error, found a class ID of " + str(id))
                print("(Filename = " + str(file_name) + ")")
                print("(Dirname = " + str(dir_name) + ")")
                x = 5 / 0
        class_id = new_class_id

        # Remove the extension (to be compatible with what Greg did...)
        file_name = file_name.replace('.fits', '')

        # Put the two together via '_' (again, to be compatible...)
        path_name = dir_name + "_" + file_name

        # Add to the sequence
        sequence_images.append(image)
        sequence_filenames.append(path_name.encode())
        sequence_classes.append(class_id)
        sequence_ymin.append(y_min)
        sequence_ymax.append(y_max)
        sequence_xmin.append(x_min)
        sequence_xmax.append(x_max)

    # Merge the 6 frames together into a video clip
    total_image = np.stack(sequence_images, axis=0)
    classes_array = np.array(pad_arrays_to_same_len(sequence_classes))
    ymin_array = np.array(pad_arrays_to_same_len(sequence_ymin))
    ymax_array = np.array(pad_arrays_to_same_len(sequence_ymax))
    xmin_array = np.array(pad_arrays_to_same_len(sequence_xmin))
    xmax_array = np.array(pad_arrays_to_same_len(sequence_xmax))

    # Create the features for this example
    features = {
        "images_raw": _bytes_feature([total_image.tostring()]),
        "filename": _bytes_feature(sequence_filenames),
        "height": _int64_feature([annotations["sensor"]["height"]]),
        "width": _int64_feature([annotations["sensor"]["width"]]),
        "num_time_steps": _int64_feature([6]),
        "classes": _bytes_feature([classes_array.tostring()]),
        "ymin": _bytes_feature([ymin_array.tostring()]),
        "ymax": _bytes_feature([ymax_array.tostring()]),
        "xmin": _bytes_feature([xmin_array.tostring()]),
        "xmax": _bytes_feature([xmax_array.tostring()]),
        "classes_shape": _int64_feature(list(classes_array.shape)),
        "ymin_shape": _int64_feature(list(ymin_array.shape)),
        "ymax_shape": _int64_feature(list(ymax_array.shape)),
        "xmin_shape": _int64_feature(list(xmin_array.shape)),
        "xmax_shape": _int64_feature(list(xmax_array.shape)),
    }

    # Create an example protocol buffer
    output_example = tf.train.Example(
        features=tf.train.Features(feature=features)
    )
    return output_example


def group_list(ungrouped_list, group_size, padding=None):

    # Magic, probably.
    grouped_list = zip_longest(*[iter(ungrouped_list)] * group_size,
                               fillvalue=padding)

    return(grouped_list)


def make_clean_dir(directory):

    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def get_immediate_subdirectories(a_dir):
    """
    Shift+CV from SO
    """

    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def build_satsim_dataset(datapath):

    # Get subdirectories of this path.

    # Iterate over each subdirectory, each of which is a collect.
    examples = list()

    for directory_name in get_immediate_subdirectories(datapath):

        print(directory_name)

        colleciton_path = os.path.join(datapath, directory_name)
        image_dir_path = os.path.join(colleciton_path, "ImageFiles")
        annotation_dir_path = os.path.join(colleciton_path, "Annotations")

        image_paths = sorted(os.listdir(image_dir_path))
        annotation_paths = sorted(os.listdir(annotation_dir_path))

        for (image_path,
             annotation_path) in zip(image_paths, annotation_paths):

            # Get first image and annothation and write to file path.
            example = (os.path.join(image_dir_path, image_path),
                       os.path.join(annotation_dir_path, annotation_path))

            examples.append(example)

    return(examples)


def build_satsim_multiframe_dataset(datapath):

    # Get subdirectories of this path.

    # Iterate over each subdirectory, each of which is a collect.
    examples = list()

    for directory_name in get_immediate_subdirectories(datapath):

        print(directory_name)

        colleciton_path = os.path.join(datapath, directory_name)
        image_dir_path = os.path.join(colleciton_path, "ImageFiles")
        annotation_dir_path = os.path.join(colleciton_path, "Annotations")

        image_paths = sorted(os.listdir(image_dir_path))
        annotation_paths = sorted(os.listdir(annotation_dir_path))

        examples_dict = {}
        for (image_path,
             annotation_path) in zip(image_paths, annotation_paths):

            # Parse the annotation JSON file
            fp = open(os.path.join(annotation_dir_path, annotation_path), "r")
            annotations = json.load(fp)["data"]
            fp.close()

            sequence = [file.encode() for file in annotations["file"]["sequence"]]
            sequence_id = annotations["file"]["sequence_id"]
            dir_name = annotations["file"]["dirname"]

            example_name = os.path.join(dir_name.encode(), sequence[0])
            if example_name not in examples_dict:
                # We've never seen this one before, so add the structure
                blank_list = []
                for i in range(len(sequence)):
                    blank_list.append('')
                examples_dict[example_name] = blank_list

            curr_sequence = examples_dict[example_name]
            curr_sequence[sequence_id] = (
                os.path.join(image_dir_path, image_path),
                os.path.join(annotation_dir_path, annotation_path)
            )

            if '' not in examples_dict[example_name]:
                examples.append(examples_dict.pop(example_name))

    return examples


def partition_examples(examples, splits_dict):

    # Create a dict to hold examples.
    partitions = dict()

    # Store the total number of examples.
    num_examples = len(examples)

    # Iterate over the items specifying the partitions.
    for (split_name, split_fraction) in splits_dict.items():

        # Compute the sixe of this parition.
        num_split_examples = int(split_fraction * num_examples)

        # Pop the next partition elements.
        partition_examples = examples[:num_split_examples]
        examples = examples[num_split_examples:]

        # Map this paritions list of examples to this parition name.
        partitions[split_name] = partition_examples

    return(partitions)


def create_tfrecords(data_dir,
                     output_dir,
                     tfrecords_name="tfrecords",
                     examples_per_tfrecord=1,
                     datapath_to_examples_fn=build_satsim_dataset,
                     tf_example_builder_fn=build_satnet_tf_example,
                     partition_examples_fn=partition_examples,
                     splits_dict={"data": 1.0}):
    """
    Given an input data directory, process that directory into examples. Group
    those examples into groups to write to a dir.
    """

    # TODO: Throw exception if interface functions aren't given.

    # Map the provided data directory to a list of tf.Examples.
    examples = datapath_to_examples_fn(data_dir)

    # Use the provided split dictionary to parition the example as a dict.
    # TODO: Interface here.
    partitioned_examples = partition_examples_fn(examples, splits_dict)

    # Iterate over each partition building the TFRecords.
    for (split_name, split_examples) in partitioned_examples.items():

        print("Writing partition %s w/ %d examples." % (split_name,
                                                        len(split_examples)))

        # Build a clean directory to store this partitions TFRecords.
        partition_output_dir = os.path.join(output_dir, split_name)
        make_clean_dir(partition_output_dir)

        # Group the examples in this paritions to write to seperate TFRecords.
        example_groups = group_list(split_examples, examples_per_tfrecord)

        # Iterate over each group. Each is a list of examples.
        for group_index, example_group in enumerate(example_groups):

            print("Saving group %s w/ %d examples" % (str(group_index),
                                                      len(example_group)))

            # Specify the group name.
            group_tfrecords_name = tfrecords_name + '_' + split_name + '_' + str(group_index) + '.tfrecords'

            # Build the path to write the output to.
            output_path = os.path.join(partition_output_dir,
                                       group_tfrecords_name)

            # Open a writer to the provided TFRecords output location.
            with tf.io.TFRecordWriter(output_path) as writer:

                # For each example...
                for example in example_group:

                    # ...if the example isn't empty...
                    if example:

                        # print("Writing example %s" % example[0])

                        # ...construct a TF Example object...
                        tf_example = tf_example_builder_fn(example)

                        # ...and write it to the TFRecord.
                        writer.write(tf_example.SerializeToString())


def get_dir_content_paths(directory):
    """
    Given a directory, returns a list of complete paths to its contents.
    """
    return([os.path.join(directory, f) for f in os.listdir(directory)])


def main(unused_argv):

    split_dict = {"train": 0.6, "valid": 0.2, "test": 0.2}

    if FLAGS.multiframe:
        datapath_fn = build_satsim_multiframe_dataset
        example_builder_fn = build_satnet_multiframe_tf_example
    else:
        datapath_fn = build_satsim_dataset
        example_builder_fn = build_satnet_tf_example

    if FLAGS.data_bank:

        for file in get_immediate_subdirectories(FLAGS.data_dir):

            input_dir = os.path.join(FLAGS.data_dir, file)
            output_dir = os.path.join(FLAGS.output_dir, "tfrecords", file)

            make_clean_dir(output_dir)

            create_tfrecords(data_dir=input_dir,
                             output_dir=output_dir,
                             examples_per_tfrecord=FLAGS.examples_per_tfrecord,
                             tfrecords_name=FLAGS.name,
                             datapath_to_examples_fn=datapath_fn,
                             tf_example_builder_fn=example_builder_fn,
                             partition_examples_fn=partition_examples,
                             splits_dict=split_dict)

    else:

        create_tfrecords(data_dir=FLAGS.data_dir,
                         output_dir=FLAGS.output_dir,
                         tfrecords_name=FLAGS.name,
                         examples_per_tfrecord=FLAGS.examples_per_tfrecord,
                         datapath_to_examples_fn=datapath_fn,
                         tf_example_builder_fn=example_builder_fn,
                         partition_examples_fn=partition_examples,
                         splits_dict=split_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str,
                        default="satsim",
                        help='Name of the dataset to build.')

    parser.add_argument('--data_dir', type=str,
                        default="/home/jfletcher/data/satnet_v2_sensor_generalization/sensor_bank/0/sensor_0",
                        help='Path to SatSim output data.')

    parser.add_argument('--output_dir', type=str,
                        default="/home/jfletcher/data/satnet_v2_sensor_generalization/sensor_bank/tfrecords/0",
                        help='Path to the output directory.')

    parser.add_argument("--examples_per_tfrecord",
                        type=int,
                        default=512,
                        help="Maximum number of examples to write to a file")

    parser.add_argument("--data_bank",
                        action='store_true',
                        default=False,
                        help="If true, make tfrecords for data_dir subdirs")

    parser.add_argument("--multiframe",
                        action='store_true',
                        default=False,
                        help="If true, make tfrecords multiframe per example")

    FLAGS, unparsed = parser.parse_known_args()

    main(unparsed)
