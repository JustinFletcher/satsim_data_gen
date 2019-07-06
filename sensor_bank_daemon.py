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


def launch_satsim_run(output_dir,
                      config_file_path,
                      device=0,
                      debug_satsim=True):

    if debug_satsim:

        cmd_str = "satsim --debug DEBUG run --device " + str(device) + " --mode eager --output_dir " + output_dir + " " + config_file_path

    else:

        cmd_str = "satsim run --device " + str(device) + " --mode eager --output_dir " + output_dir + " " + config_file_path

    process = subprocess.Popen(cmd_str,
                               shell=True,
                               stdout=subprocess.PIPE)
    process.wait()
    print(process.returncode)
    return(process.returncode)


def populate_needed_samples(needed_samples_dict,
                            device=0,
                            debug_satsim=True,
                            min_elements=1,
                            fill_fraction=1.0):

    for sensor_dir_path, num_elements_needed in needed_samples_dict.items():

        print("Sensor at " + sensor_dir_path + " needs " + str(num_elements_needed))

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
            num_elements_needed = int(np.max([num_elements_needed,
                                              min_elements]))

            print("I'm going to produce " + str(num_elements_needed))

            sensor_config_dict["sim"]["samples"] = num_elements_needed

            print(sensor_config_dict)

            # Write a JSON file in the new dir.
            with open(sensor_config_file_path, 'w') as fp:

                json.dump(sensor_config_dict, fp)

            launch_satsim_run(output_dir=sensor_dir_path,
                              config_file_path=sensor_config_file_path,
                              device=device,
                              debug_satsim=FLAGS.debug_satsim)


def main(**kwargs):

    # This is a daemon, so it's gonna run until you kill it..
    while True:

        # Build a dict mapping sensor paths to number of samples needed.
        needed_samples_dict = compute_needed_samples(FLAGS.data_bank_dir,
                                                     FLAGS.num_samples)

        # print(needed_samples_dict)

        # So long as there are sensors that need completion, complete them.
        while sum(np.array(list(needed_samples_dict.values())).clip(0)) > 0:

            # Build a dict mapping sensor paths to number of samples needed.
            needed_samples_dict = compute_needed_samples(FLAGS.data_bank_dir,
                                                         FLAGS.num_samples)

            # Then, populate some or all those samples.
            populate_needed_samples(needed_samples_dict,
                                    device=FLAGS.device,
                                    min_elements=FLAGS.min_elements,
                                    debug_satsim=FLAGS.debug_satsim,
                                    fill_fraction=FLAGS.fill_fraction)

        # Once here, all the sensors have been filled; make a new one...
        with open(FLAGS.config_file_path, 'r') as f:

            # Read the base config file which randomizes over other properties.
            base_config_dict = json.load(f)

        # Build a new randomized config dict for this sensor;
        config_dict = build_sensor_config(1,
                                          FLAGS.num_frames_per_sample,
                                          base_config_dict)

        # Create a directory for this sensors examples; first get existing.
        sensor_dirs = glob.glob(os.path.join(FLAGS.data_bank_dir, "sensor_*"))

        # Then take the suffix post "sensor_", int it, max it, and add 1.
        sensor_nums = [int(s.split("sensor_")[-1]) for s in sensor_dirs]
        sensor_num = np.max(sensor_nums) + 1

        # construct the new sensor name, and make a dir name to hold its data.
        sensor_name_str = "sensor_" + str(sensor_num)
        sensor_dir_path = os.path.join(FLAGS.data_bank_dir, sensor_name_str)

        print("Making: " + sensor_name_str)

        # Clear this sensor dir if it exists, then make it.
        make_clean_dir(sensor_dir_path)

        # Build a filename for this config.
        sensor_json_file = "sensor_" + str(sensor_num) + ".json"
        output_config_file = os.path.join(sensor_dir_path, sensor_json_file)

        # Write a JSON file in the new dir.
        with open(output_config_file, 'w') as fp:

            json.dump(config_dict, fp)


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

    parser.add_argument('--num_samples', type=int,
                        default=16,
                        help='The number of samples from each sensor.')

    parser.add_argument('--num_frames_per_sample', type=int,
                        default=6,
                        help='The number of frames to use in each sequence.')

    parser.add_argument('--device', type=int,
                        default=0,
                        help='Number of the GPU use.')

    parser.add_argument('--fill_fraction', type=float,
                        default=0.1,
                        help='A float in (0, 1] tunes sample production goal.')

    parser.add_argument('--min_elements', type=int,
                        default=1,
                        help='An int determining the smallest run size.')

    parser.add_argument('--debug_satsim', action='store_true',
                        default=False,
                        help='If true, write annotated JPEGs to disk.')

    FLAGS = parser.parse_args()

    main()
