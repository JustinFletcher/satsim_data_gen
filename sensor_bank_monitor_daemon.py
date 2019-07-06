"""
author: Justin Fletcher
date: 20 Jun 2019
"""


import os

import time

import json

import argparse

import tensorflow as tf


def get_immediate_subdirectories(a_dir):
    """
    Shift+CV from SO
    """

    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def compute_samples_counts(data_bank_dir):

    sample_counts = dict()

    # First, read the sensor_* subdirectories of the given directory.
    for sensor_dir in get_immediate_subdirectories(data_bank_dir):

        sensor_dir_path = os.path.join(data_bank_dir, sensor_dir)

        data_element_dir_list = get_immediate_subdirectories(sensor_dir_path)

        num_elements_produced = len(data_element_dir_list)

        sample_counts[sensor_dir_path] = num_elements_produced

    return(sample_counts)


def main(**kwargs):

    if FLAGS.watch:

        # This is a daemon, so it's gonna run until you kill it..
        while True:

            # Build a dict mapping sensor paths to number of samples produced.
            samples_dict = compute_samples_counts(FLAGS.data_bank_dir)

            print(json.dumps(samples_dict, indent=1))

        time.sleep(FLAGS.n)

    else:

        # Build a dict mapping sensor paths to number of samples produced.
        samples_dict = compute_samples_counts(FLAGS.data_bank_dir)

        print(json.dumps(samples_dict, indent=1))


if __name__ == '__main__':

    print(tf.__version__)

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_bank_dir', type=str,
                        default="/home/jfletcher/data/satnet_v2_sensor_generalization/sensor_bank_debug/",
                        help='Path to SatSim sensor bank.')

    parser.add_argument('--n', type=float,
                        default=1.0,
                        help="Float dictating update rate.")

    parser.add_argument('--watch', action='store_true',
                        default=False,
                        help="If true, use python to print output.")

    FLAGS = parser.parse_args()

    main()
