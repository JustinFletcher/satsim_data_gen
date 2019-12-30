"""
author: Justin Fletcher
date: 20 Jun 2019
"""

import os

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

    geometry_dict = config_dict['geometry']
    geometry_obs_dict = geometry_dict['obs']
    obs_list_dict = geometry_obs_dict['list']
    # This is a list of 2 dicts.
    obs_list_origin_list = obs_list_dict['origin']

    spread_limit = .7

    x_orig_center = np.random.uniform(spread_limit, 1 - spread_limit)
    obs_list_origin_list[0]['low'] = x_orig_center + np.random.uniform(
        -spread_limit, spread_limit)
    obs_list_origin_list[0]['high'] = x_orig_center + np.random.uniform(
        -spread_limit, spread_limit)

    y_orig_center = np.random.uniform(spread_limit, 1 - spread_limit)
    obs_list_origin_list[1]['low'] = y_orig_center + np.random.uniform(
        -spread_limit, spread_limit)
    obs_list_origin_list[1]['high'] = y_orig_center + np.random.uniform(
        -spread_limit, spread_limit)

    obs_list_dict['origin'] = obs_list_origin_list
    geometry_obs_dict['list'] = obs_list_dict
    geometry_dict['obs'] = geometry_obs_dict
    config_dict['geometry'] = geometry_dict

    return (config_dict)


def build_chaff_config(base_config_dict):
    config_dict = base_config_dict

    return config_dict


def make_clean_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def main(FLAGS):
    cmd_strings = list()

    # Generate a new config file randomly selecting from an FPA config.
    for sensor_num in range(FLAGS.num_sensors):

        with open(FLAGS.config_file_path, 'r') as f:

            # Read the base config file which randomizes over other properties.
            base_config_dict = json.load(f)

        config_dict = build_sensor_config(FLAGS.num_samples,
                                          FLAGS.num_frames_per_sample,
                                          base_config_dict)

        # Create a directory for this sensors runs.
        sensor_name_str = "sensor_" + str(sensor_num)
        sensor_dir = os.path.join(FLAGS.output_dir, sensor_name_str)

        # Clear this sensor dir if it exists, then make it.
        make_clean_dir(sensor_dir)

        # Build a filename for this config.
        sensor_json_file = "sensor_" + str(sensor_num) + ".json"
        output_config_file = os.path.join(sensor_dir, sensor_json_file)

        # Write a JSON file in the new dir.
        with open(output_config_file, 'w') as fp:

            json.dump(config_dict, fp)

        if FLAGS.debug_satsim:

            cmd_str = "satsim --debug DEBUG run --device " + str(
                FLAGS.device) + " --mode eager --output_dir " + sensor_dir +\
                      " " + output_config_file

        else:

            cmd_str = "satsim run --device " + str(
                FLAGS.device) + " --mode eager --output_dir " + sensor_dir +\
                      " " + output_config_file

        cmd_strings.append(cmd_str)

    # Iterate over the commands...
    for cmd_str in cmd_strings:
        # ...sequentially launching each.
        print(cmd_str)
        process = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE)
        process.wait()
        print(process.returncode)


if __name__ == '__main__':
    print(tf.__version__)

    parser = argparse.ArgumentParser()

    # Set arguments and their default values
    # parser.add_argument('--config_file_path', type=str,
    #                     default="/home/jfletcher/research/satsim_data_gen
    #                     /satsim.json",
    #                     help='Path to the JSON config for SatSim.')
    #
    # parser.add_argument('--output_dir', type=str,
    #                     default="/home/jfletcher/data/satsim_data_gen/",
    #                     help='Path to the JSON config for SatSim.')

    parser.add_argument('--config_file_path', type=str,
                        default="C:\\research\\satsim_data_gen\\satsim_cso"
                                ".json",
                        help='Path to the JSON config for SatSim.')

    parser.add_argument('--output_dir', type=str,
                        default=" C:\\research\\satsim_data_gen\\temp",
                        help='Path to the JSON config for SatSim.')

    parser.add_argument('--num_sensors', type=int,
                        default=10000,
                        help='The number of sensors to simulate.')

    parser.add_argument('--num_samples', type=int,
                        default=1,
                        help='The number of samples from each sensor.')

    parser.add_argument('--num_frames_per_sample', type=int,
                        default=1,
                        help='The number of samples from each sensor.')

    parser.add_argument('--device', type=int,
                        default=0,
                        help='Number of the GPU use.')

    parser.add_argument('--debug_satsim', action='store_true',
                        default=False,
                        help='If true, write annotated JPEGs to disk.')

    FLAGS = parser.parse_args()

    main(FLAGS)
