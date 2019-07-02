"""
author: Justin Fletcher
date: 20 Jun 2019
"""


import os

import glob

import json

import shutil

import argparse

import subprocess

import numpy as np

import tensorflow as tf


def build_sensor_config(num_samples, num_frames_per_sample, base_config_dict):

    config_dict = base_config_dict

    sim_dict = config_dict['sim']
    sim_dict["samples"] = num_samples
    config_dict['sim'] = sim_dict

    fpa_dict = config_dict['fpa']

    # Select a FOV for this sensor.
    fov = np.random.uniform(0.3, 1.5)
    fpa_dict["y_fov"] = fov
    fpa_dict["x_fov"] = fov

    # Dark current in photoelectrons. Consider Lognormal around about 1.
    fpa_dict["dark_current"] = np.random.uniform(0.5, 5)

    # Gain and bias.
    fpa_dict["gain"] = np.random.uniform(1.0, 2.0)
    fpa_dict["bias"] = np.random.uniform(90, 110)
    fpa_dict["zeropoint"] = np.random.uniform(21.0, 26.0)

    # fpa_dict = config_dict['fpa']
    # a2d_dict = fpa_dict['a2d']
    a2d_dict = dict()
    a2d_dict["response"] = "linear"
    a2d_dict["fwc"] = np.random.uniform(190000, 200000)
    a2d_dict["gain"] = np.random.uniform(1.0, 2.0)
    a2d_dict["bias"] = np.random.uniform(9, 11)
    fpa_dict["a2d"] = a2d_dict

    # Read noise for smae sensor.
    # noise_dict = fpa_dict["noise"]
    noise_dict = dict()
    noise_dict["read"] = np.random.uniform(5, 20)
    noise_dict["electronic"] = np.random.uniform(5, 10)
    fpa_dict["noise"] = noise_dict

    # psf_dict = fpa_dict["psf"]
    psf_dict = dict()
    psf_dict["mode"] = "gaussian"
    psf_dict["eod"] = np.random.uniform(0.05, 0.9)
    fpa_dict["psf"] = psf_dict

    # time_dict = fpa_dict["time"]
    time_dict = dict()
    time_dict["exposure"] = np.random.uniform(1.0, 2.0)
    time_dict["gap"] = np.random.uniform(0.1, 1)
    fpa_dict["time"] = time_dict

    fpa_dict["num_frames"] = num_frames_per_sample

    # Set the FPA.
    config_dict['fpa'] = fpa_dict

    return(config_dict)


def make_clean_dir(directory):

    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def get_immediate_subdirectories(a_dir):
    """
    Shift+CV from SO
    """

    return [name for name in os.listdir(a_dir)

            if os.path.isdir(os.path.join(a_dir, name))]


def compute_needed_samples(data_bank_dir, num_samples):

    needed_samples = dict()

    # First, read the sensor_* subdirectories of the given directory.
    for sensor_dir in get_immediate_subdirectories(data_bank_dir):

        sensor_dir_path = os.path.join(data_bank_dir, sensor_dir)

        data_element_dir_list = get_immediate_subdirectories(sensor_dir_path)

        num_elements_produced = len(data_element_dir_list)

        num_elements_needed = num_samples - num_elements_produced

        needed_samples[sensor_dir_path] = num_elements_needed

    return(needed_samples)


def populate_needed_samples(needed_samples_dict,
                            device=0,
                            fill_fraction=1.0):

    for sensor_dir_path, num_elements_needed in needed_samples_dict.items():

        # Check if more examples are needed form this sensor; if so, make them.
        if num_elements_needed > 0:

            # First, glob onto all files matching *.json in this sensor dir.
            sensor_config_pattern = os.path.join(sensor_dir_path, "*.json")
            sensor_config_file_paths = glob.glob(sensor_config_pattern)

            # Ensure that one path is returned, otherwise something's wrong...
            if len(sensor_config_file_paths) != 1:

                # ...so stop here to avoid damage.
                print(sensor_config_file_paths)
                raise Exception('Found >1 *.json in ' + sensor_dir_path)

            # But, if nothing is wrong...
            else:

                # s...strip the list, because we know there's only one element.
                sensor_config_file_path = sensor_config_file_paths[0]

            print(sensor_config_file_path)

            # Open the single JSON file configuring this sensor...
            with open(sensor_config_file_path, 'r') as f:

                # ...and read its contents to a dict.
                sensor_config_dict = json.load(f)

            # Select a fraction os the needed samples to make; make at least 1.
            num_elements_needed = int(num_elements_needed * fill_fraction)
            num_elements_needed = int(np.max([num_elements_needed, 1.0]))

            print("I'm going to produce " + str(num_elements_needed))

            sensor_config_dict["samples"] = num_elements_needed

            # Write a JSON file in the new dir.
            with open(sensor_config_file_path, 'w') as fp:

                json.dump(sensor_config_dict, fp)

            if FLAGS.debug_satsim:

                cmd_str = "satsim --debug DEBUG run --device " + str(device) + " --mode eager --output_dir " + sensor_dir_path + " " + sensor_config_file_path

            else:

                cmd_str = "satsim run --device " + str(device) + " --mode eager --output_dir " + sensor_dir_path + " " + sensor_config_file_path

            process = subprocess.Popen(cmd_str,
                                       shell=True,
                                       stdout=subprocess.PIPE)
            process.wait()
            print(process.returncode)

            # print("I would have run: " + cmd_str)


def main(**kwargs):

    # This is a daemon, so it's gonna run until you kill it..
    while True:

        # Build a dict mapping sensor paths to number of samples needed.
        needed_samples_dict = compute_needed_samples(FLAGS.data_bank_dir,
                                                     FLAGS.num_samples)

        # So long as there are sensors that need completion, complete them.
        while sum(needed_samples_dict.values()) > 0:

            # Build a dict mapping sensor paths to number of samples needed.
            needed_samples_dict = compute_needed_samples(FLAGS.data_bank_dir,
                                                         FLAGS.num_samples)

            print(needed_samples_dict)

            # Then, populate some or all those samples.
            populate_needed_samples(needed_samples_dict,
                                    device=FLAGS.device,
                                    fill_fraction=0.1)

        # If we're here, all the sensors have been completed; make a new one.

        die

    

    # If we're here, every existing sensor dir has the right number of samples.




if __name__ == '__main__':

    print(tf.__version__)

    parser = argparse.ArgumentParser()


    parser.add_argument('--data_bank_dir', type=str,
                        default="/home/jfletcher/data/satnet_v2_sensor_generalization/sensor_bank_debug/",
                        help='Path to SatSim sensor bank.')


    # Set arguments and their default values
    parser.add_argument('--config_file_path', type=str,
                        default="/home/jfletcher/research/satsim_data_gen/satsim.json",
                        help='Path to the JSON config for SatSim.')

    parser.add_argument('--sensor_bank_dir', type=str,
                        default="/home/jfletcher/data/satsim_data_gen/",
                        help='Path to the JSON config for SatSim.')

    # parser.add_argument('--config_file_path', type=str,
    #                     default="C:\\research\\satsim_data_gen\\sensor_data_config.json",
    #                     help='Path to the JSON config for SatSim.')

    # parser.add_argument('--output_dir', type=str,
    #                     default="C:\\data\\satsim_data_gen\\",
    #                     help='Path to the JSON config for SatSim.')

    parser.add_argument('--num_samples', type=int,
                        default=16,
                        help='The number of samples from each sensor.')

    parser.add_argument('--num_frames_per_sample', type=int,
                        default=6,
                        help='The number of frames to use in each sequence.')

    parser.add_argument('--device', type=int,
                        default=0,
                        help='Number of the GPU use.')

    parser.add_argument('--debug_satsim', action='store_true',
                        default=False,
                        help='If true, write annotated JPEGs to disk.')

    FLAGS = parser.parse_args()

    main()
