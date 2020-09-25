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
def layout_generate(general_para):
    N = general_para.n_links
    # first, generate transmitters' coordinates
    tx_xs = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
    tx_ys = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
    layout_rx = []
    # generate rx one by one rather than N together to ensure checking validity one by one
    rx_xs = []; rx_ys = []
    tot_links = 0
    n_re = general_para.n_receiver
    for i in range(N):
        n_links = i
        rx_i = []

        num_rx = np.random.randint(general_para.minrx, general_para.maxrx)
        num_rx = min(num_rx,  n_re - tot_links)
        tot_links += num_rx
        for j in range(num_rx): 
            got_valid_rx = False
            while(not got_valid_rx):
                pair_dist = np.random.uniform(low=general_para.shortest_directLink_length, high=general_para.longest_directLink_length)
                pair_angles = np.random.uniform(low=0, high=np.pi*2)
                rx_x = tx_xs[i] + pair_dist * np.cos(pair_angles)
                rx_y = tx_ys[i] + pair_dist * np.sin(pair_angles)
                if(0<=rx_x<=general_para.field_length and 0<=rx_y<=general_para.field_length):
                    got_valid_rx = True
            rx_i.append([rx_x[0], rx_y[0]])
        layout_rx.append(rx_i)
        if(tot_links >= n_re):
            break

    # For now, assuming equal weights and equal power, so not generating them
    layout_tx = np.concatenate((tx_xs, tx_ys), axis=1)
    
    return layout_tx, layout_rx

def distance_generate(general_para,layout_tx,layout_rx):
    distances = np.zeros((general_para.n_receiver,general_para.n_receiver))
    N = len(layout_rx)
    sum_tx = 0
    for tx_index in range(N):
        num_loops = len(layout_rx[tx_index])
        tx_coor = layout_tx[tx_index]
        for tx_i in range(num_loops):
            sum_rx = 0
            for rx_index in range(N):
                for rx_i in layout_rx[rx_index]:
                    rx_coor = rx_i
                    distances[sum_rx][sum_tx] = np.linalg.norm(tx_coor - rx_coor)
                    sum_rx += 1
            sum_tx += 1
    return distances
def CSI_generate(general_para, distances):
    Nt = general_para.N_antennas
    L = general_para.n_receiver
    dists = np.expand_dims(distances,axis=-1)
    shadowing = np.random.randn(L,L,Nt)
    large_scale_CSI = 4.4*10**5/((dists**1.88)*(10**(shadowing*6.3/20)))
    small_scale_CSI = 1/np.sqrt(2)*(np.random.randn(L,L,Nt)+1j*np.random.randn(L,L,Nt))*np.sqrt(large_scale_CSI)
    return small_scale_CSI
def sample_generate(general_para, number_of_layouts, norm = None):
    print("<<<<<<<<<<<<<{} layouts: {}>>>>>>>>>>>>".format(
        number_of_layouts, general_para.setting_str))
    layouts = []
    dists = []
    CSIs = []
    for i in range(number_of_layouts):
        # generate layouts
        layout_tx, layout_rx = layout_generate(general_para)
        n_re = general_para.n_receiver
        dis = distance_generate(general_para,layout_tx,layout_rx)
        csis = CSI_generate(general_para, dis)
        
        #data collection
        dists.append(dis)
        CSIs.append(csis)
            
    dists = np.array(dists)
    CSIs = np.array(CSIs)
    return dists, CSIs
