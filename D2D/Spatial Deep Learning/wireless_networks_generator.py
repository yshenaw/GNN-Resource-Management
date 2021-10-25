# This script contains the generator code for producing wireless network layouts and channel losses
# for the work "Spatial Deep Learning for Wireless Scheduling",
# available at https://ieeexplore.ieee.org/document/8664604.

# For any reproduce, further research or development, please kindly cite our JSAC journal paper:
# @Article{spatial_learn,
#    author = "W. Cui and K. Shen and W. Yu",
#    title = "Spatial Deep Learning for Wireless Scheduling",
#    journal = "{\it IEEE J. Sel. Areas Commun.}",
#    year = 2019,
#    volume = 37,
#    issue = 6,
#    pages = "1248-1261",
#    month = "June",
# }

import numpy as np
#import general_parameters
import helper_functions

def generate_layouts(general_para, number_of_layouts):
    N = general_para.n_links
    print("<<<<<<<<<<<<<{} layouts: {}>>>>>>>>>>>>".format(
        number_of_layouts, general_para.setting_str))
    layouts = []
    dists = []
    for i in range(number_of_layouts):
        layout, dist = helper_functions.layout_generate(general_para)
        layouts.append(layout)
        dists.append(dist)
    layouts = np.array(layouts)
    dists = np.array(dists)
    assert np.shape(layouts)==(number_of_layouts, N, 4)
    assert np.shape(dists)==(number_of_layouts, N, N)
    return layouts, dists

# compute path loss components of channel path_losses
# should be used with multiple layouts:
#        distances shape: number of layouts X N X N
def compute_path_losses(general_para, distances):
    N = np.shape(distances)[-1]
    assert N==general_para.n_links

    h1 = general_para.tx_height
    h2 = general_para.rx_height
    signal_lambda = 2.998e8 / general_para.carrier_f
    antenna_gain_decibel = general_para.antenna_gain_decibel
    # compute relevant quantity
    Rbp = 4 * h1 * h2 / signal_lambda
    Lbp = abs(20 * np.log10(np.power(signal_lambda, 2) / (8 * np.pi * h1 * h2)))
    # compute coefficient matrix for each Tx/Rx pair
    sum_term = 20 * np.log10(distances / Rbp)
    Tx_over_Rx = Lbp + 6 + sum_term + ((distances > Rbp).astype(int)) * sum_term  # adjust for longer path loss
    pathlosses = -Tx_over_Rx + np.eye(N) * antenna_gain_decibel  # only add antenna gain for direct channel
    pathlosses = np.power(10, (pathlosses / 10))  # convert from decibel to absolute
    return pathlosses
