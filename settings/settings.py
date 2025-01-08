import configparser
import sys
import numpy as np
import argparse
import os
import shutil

def read_parameters():
    args = read_cmdline_args()
    #file_path = sys.argv[1]
    file_path = args.parameter

    config = configparser.ConfigParser()
    
    # Use delimiters to ignore the comment section after the semicolon
    config.optionxform = str  # Preserve case-sensitivity of keys
    config.read_string('[DEFAULT]\n' + open(file_path).read())  # Prepend a dummy section header

    parameters = {key: value.split(';')[0].strip() for key, value in config['DEFAULT'].items()}

    return parameters

def read_cmdline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outdir", help="name of output directory", default="simulation_results")
    parser.add_argument("-init", "--initialization", help="specify initial condition",choices=['4dots','random-dots','white-noise'])
    parser.add_argument("-vid","--videomode", help="more plots in between", action="store_true")
    parser.add_argument("-dimless","--dimensionless", help="dimensionless?", action="store_true")
    parser.add_argument("-p", "--parameter", help="parameter file")
    parser.add_argument("-i", "--input", nargs=3, help="path to starting solution")
    parser.add_argument("-m", "--model", help="specify model", default="NL", choices=["NL","GM"])
    parser.add_argument("-tdisc", "--timedisc", help="specify temporal discretization method", default="EE", choices=["EE", "H", "strang_EE_IE", "strang_H_CN"])
    parser.add_argument("-d", "--dimensions", help="number of spatial dimensions", default=2)
    args = parser.parse_args()
    return args


def print_settings(parameters):
    for param, value in parameters.items():
        print(f"{param} = {value}")

def setup_output_directory(dirname):
    """Set up output directory structure"""
    if not os.path.exists(f"out/{dirname}"):
        os.makedirs(f"out/{dirname}/plots")
        os.makedirs(f"out/{dirname}/data")
    else:
        shutil.rmtree(f"out/{dirname}")
        print(f"Old output directory '{dirname}' deleted")
        os.makedirs(f"out/{dirname}/plots")
        os.makedirs(f"out/{dirname}/data")