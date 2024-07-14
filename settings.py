import configparser
import sys
import numpy as np

def read_parameters():
    file_path = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(file_path)

    parameters = {}
    for section in config.sections():
        for key, value in config.items(section):
            parameters[key] = value

    alpha_N = float(parameters.get('alpha_n'))
    alpha_L = float(parameters.get('alpha_l'))
    k_p = float(parameters.get('k_p'))
    n_N = float(parameters.get('n_n'))
    n_L = float(parameters.get('n_l'))
    K_N = float(parameters.get('k_n'))
    K_L = float(parameters.get('k_l'))
    gamma_N = float(parameters.get('gamma_n'))
    gamma_L = float(parameters.get('gamma_l'))
    D_N = float(parameters.get('d_n'))
    D_L = float(parameters.get('d_l'))

    return alpha_N, alpha_L, k_p, n_N, n_L, K_N, K_L, gamma_N, gamma_L, D_N, D_L

def print_settings():
    alpha_N, alpha_L, k_p, n_N, n_L, K_N, K_L, gamma_N, gamma_L, D_N, D_L = read_parameters()
    print("### parameters ###")
    print("alpha_N =", alpha_N)
    print("alpha_L =", alpha_L)
    print("k_p =", k_p)
    print("n_N =", n_N)
    print("n_L =", n_L)
    print("K_N =", K_N)
    print("K_L =", K_L)
    print("gamma_N =", gamma_N)
    print("gamma_L =", gamma_L)
    print("D_N =", D_N)
    print("D_L =", D_L)
    print(" ")