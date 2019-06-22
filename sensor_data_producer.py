

import json

import argparse

import subprocess

import numpy as np

import tensorflow as tf


def main(**kwargs):

    print(FLAGS.config_file_path)

    # Read the base config file which randomizes over star and sat properties.
    config_dict = json.loads(FLAGS.config_file_path)[0]

    # Generate a new config file randomly selecting from an FPA config
    for sensor_num in range(FLAGS.num_sensors):

        # Select a FOV for this sensor.
        fov = np.random.uniform(0.3, 1.5)
        config_dict["y_fov"] = fov
        config_dict["x_fov"] = fov

        # Dark current in photoelectrons. Consider Lognormal around about 1.
        config_dict["dark_current"] = np.random.uniform(0.5, 5)

        # Gain and bias.
        config_dict["gain"] = np.random.uniform(1.0, 2.0)
        config_dict["bias"] = np.random.uniform(90, 110)
        config_dict["zeropoint"] = np.random.uniform(21.0, 26.0)

        a2d_dict = config_dict["a2d"]
        a2d_dict["response"] = "linear"
        a2d_dict["fwc"] = np.random.uniform(190000, 200000)
        a2d_dict["gain"] = np.random.uniform(1.0, 2.0)
        a2d_dict["bias"] = np.random.uniform(9, 11)

        config_dict["a2d"] = a2d_dict

        # Read noise for smae sensor.
        noise_dict = config_dict["noise"]
        noise_dict["read"] = np.random.uniform(5, 20)
        noise_dict["electronic"] = np.random.uniform(5, 10)
        config_dict["noise"] = noise_dict

        psf_dict = config_dict["psf"]
        psf_dict["mode"] = "gaussian"
        psf_dict["eod"] = np.random.uniform(0.05, 0.9)
        config_dict["psf"] = psf_dict

        time_dict = config_dict["time"]

        time_dict["exposure"] = np.random.uniform(1.0, 2.0)
        time_dict["gap"] = np.random.uniform(0.1, 1)

        config_dict["time"] = time_dict

        # Create a directory for this sensors runs.

        sensor_dir = FLAGS.output_dir + "sensor_" + str(sensor_num) + "/"

        output_config_file = sensor_dir + "sensor_" + str(sensor_num) + ".json"

        with open(output_config_file, 'w') as fp:

            json.dump(config_dict, fp)

        cmd_str = "satsim --debug DEBUG run --device " + FLAGS.device + " --mode eager --output_dir " + sensor_dir + " " + output_config_file

        subprocess.run(cmd_str)


if __name__ == '__main__':

    print(tf.__version__)

    parser = argparse.ArgumentParser()

    # Set arguments and their default values
    parser.add_argument('--name', type=str,
                        default="ipnetv0",
                        help='Name of this model.')

    parser.add_argument('--config_file_path', type=str,
                        default="/home/jfletcher/research/satnet_v2_sensor_generalization/satsim.json",
                        help='Path to the JSON config for SatSim.')

    parser.add_argument('--output_dir', type=str,
                        default="/home/jfletcher/data/satnet_v2_sensor_generalization/",
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

    parser.add_argument('--num_channels', type=int,
                        default=3,
                        help='Number of channels in the input data.')

    parser.add_argument('--device', type=int,
                        default=0,
                        help='Number of the GPU use.')

    FLAGS = parser.parse_args()

    main()
