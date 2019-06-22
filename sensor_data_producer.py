
import os

import json

import shutil

import argparse

import subprocess

import numpy as np

import tensorflow as tf


def main(**kwargs):

    # print(FLAGS.config_file_path)

    with open(FLAGS.config_file_path, 'r') as f:

        # Read the base config file which randomizes over other properties.
        config_dict = json.load(f)

    cmd_strings = list()

    # Generate a new config file randomly selecting from an FPA config.
    for sensor_num in range(FLAGS.num_sensors):

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

        # Set the FPA.
        config_dict['fpa'] = fpa_dict

        # Create a directory for this sensors runs.
        sensor_dir = os.path.join(FLAGS.output_dir,
                                  "sensor_" + str(sensor_num))

        # Clear this sensor dir if it exists, then make it.
        if os.path.exists(sensor_dir):
            shutil.rmtree(sensor_dir)
        os.mkdir(sensor_dir)

        # Build a filename for this config.
        sensor_json_file = "sensor_" + str(sensor_num) + ".json"
        output_config_file = os.path.join(sensor_dir, sensor_json_file)

        # Write a JSON file in the new dir.
        with open(output_config_file, 'w') as fp:

            json.dump(config_dict, fp)

        cmd_str = "satsim --debug DEBUG run --device " + str(FLAGS.device) + " --mode eager --output_dir " + sensor_dir + " " + output_config_file

    # Iterate over the commands...
    for cmd_str in cmd_strings:

        # ...sequentially launching each.
        process = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE)
        process.wait()


if __name__ == '__main__':

    print(tf.__version__)

    parser = argparse.ArgumentParser()

    # Set arguments and their default values
    # parser.add_argument('--config_file_path', type=str,
    #                     default="/home/jfletcher/research/satnet_v2_sensor_generalization/satsim.json",
    #                     help='Path to the JSON config for SatSim.')

    # parser.add_argument('--output_dir', type=str,
    #                     default="/home/jfletcher/data/satnet_v2_sensor_generalization/",
    #                     help='Path to the JSON config for SatSim.')

    parser.add_argument('--config_file_path', type=str,
                        default="C:\\research\\satsim_data_gen\\sensor_data_config.json",
                        help='Path to the JSON config for SatSim.')

    parser.add_argument('--output_dir', type=str,
                        default="C:\\data\\satsim_data_gen\\",
                        help='Path to the JSON config for SatSim.')

    parser.add_argument('--num_sensors', type=int,
                        default=64,
                        help='The number of sensors to simulate.')

    parser.add_argument('--num_samples', type=int,
                        default=32,
                        help='The number of samples from each sensor.')

    parser.add_argument('--num_frames', type=int,
                        default=6,
                        help='The number of frames to use in each sequence.')

    parser.add_argument('--device', type=int,
                        default=0,
                        help='Number of the GPU use.')

    FLAGS = parser.parse_args()

    main()
