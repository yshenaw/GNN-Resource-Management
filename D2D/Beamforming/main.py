import scipy.io as sio                     # import scipy.io for .mat file I/O 
import numpy as np                         # import numpy
import matplotlib.pyplot as plt            # import matplotlib.pyplot for figure plotting
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
import wireless_networks_generator as wg
import wmmse


class init_parameters():
    def __init__(self):
        # wireless network settings
        self.n_links = train_K
        self.n_receiver = train_K
        self.field_length = 1000
        self.shortest_directLink_length = 2
        self.longest_directLink_length = 65
        self.shortest_crossLink_length = 1
        self.bandwidth = 5e6
        self.carrier_f = 2.4e9
        self.tx_height = 1.5
        self.rx_height = 1.5
        self.antenna_gain_decibel = 2.5
        self.tx_power_milli_decibel = 40
        self.tx_power = 1#np.power(10, (self.tx_power_milli_decibel-30)/10)
        self.noise_density_milli_decibel = -169
        self.input_noise_power = np.power(10, ((self.noise_density_milli_decibel-30)/10)) * self.bandwidth
        self.output_noise_power = 1#self.input_noise_power
        self.SNR_gap_dB = 6
        self.SNR_gap = 1#np.power(10, self.SNR_gap_dB/10)
        self.setting_str = "{}_links_{}X{}_{}_{}_length".format(self.n_links, self.field_length, self.field_length, self.shortest_directLink_length, self.longest_directLink_length)
        # 2D occupancy grid setting
        self.cell_length = 5
        self.N_antennas = Nt
        self.maxrx = 2
        self.minrx = 1
        self.n_grids = np.round(self.field_length/self.cell_length).astype(int)

def normalize_data(train_data, test_data, general_para):
    Nt = general_para.N_antennas
    
    tmp_mask = np.expand_dims(np.eye(train_K),axis=-1)
    tmp_mask = [tmp_mask for i in range(Nt)]
    mask = np.concatenate(tmp_mask,axis=-1)
    mask = np.expand_dims(mask,axis=0)
    
    train_copy = np.copy(train_data)
    diag_H = np.multiply(mask,train_copy)
    diag_mean = np.sum(diag_H/Nt)/train_layouts/train_K
    diag_var = np.sqrt(np.sum(np.square(diag_H))/train_layouts/train_K/Nt)
    tmp_diag = (diag_H - diag_mean)/diag_var

    off_diag = train_copy - diag_H
    off_diag_mean = np.sum(off_diag/Nt)/train_layouts/train_K/(train_K-1)
    off_diag_var = np.sqrt(np.sum(np.square(off_diag))/Nt/train_layouts/train_K/(train_K-1))
    tmp_off = (off_diag - off_diag_mean)/off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off,mask)
    
    norm_train = np.multiply(tmp_diag,mask) + tmp_off_diag
    
    # normlize test
    tmp_mask = np.expand_dims(np.eye(test_K),axis=-1)
    tmp_mask = [tmp_mask for i in range(Nt)]
    mask = np.concatenate(tmp_mask,axis=-1)
    mask = np.expand_dims(mask,axis=0)
    
    test_copy = np.copy(test_data)
    diag_H = np.multiply(mask,test_copy)
    tmp_diag = (diag_H - diag_mean)/diag_var
    
    off_diag = test_copy - diag_H
    tmp_off = (off_diag - off_diag_mean)/off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off,mask)
    
    norm_test = np.multiply(tmp_diag,mask) + tmp_off_diag
    print(diag_mean, diag_var, off_diag_mean, off_diag_var)
    return norm_train, norm_test

def batch_wmmse(csis,var_noise):
    Nt = test_config.N_antennas
    K = test_config.n_receiver
    n = csis.shape[0]
    Y = np.zeros( (n,K,Nt),dtype=complex)
    Pini = 1/np.sqrt(Nt)*np.ones((K,Nt),dtype=complex)
    for ii in range(n):
        Y[ii,:,:] = wmmse.np_WMMSE_vector(np.copy(Pini), csis[ii,:,:,:], 1, var_noise)
    return Y

def build_graph(CSI, dist, norm_csi_real, norm_csi_imag, K, threshold):
    n = CSI.shape[0]
    Nt = CSI.shape[2]
    x1 = np.array([CSI[ii,ii,:] for ii in range(K)])
    x2 = np.imag(x1)
    x1 = np.real(x1)
    x3 = 1/np.sqrt(Nt)*np.ones((n,2*Nt))
    
    x = np.concatenate((x3,x1,x2),axis=1)
    x = torch.tensor(x, dtype=torch.float)
    
    
    dist2 = np.copy(dist)
    mask = np.eye(K)
    diag_dist = np.multiply(mask,dist2)
    dist2 = dist2 + 1000 * diag_dist
    dist2[dist2 > threshold] = 0
    attr_ind = np.nonzero(dist2)
    
    edge_attr_real = norm_csi_real[attr_ind]
    edge_attr_imag = norm_csi_imag[attr_ind]
    
    edge_attr = np.concatenate((edge_attr_real,edge_attr_imag), axis=1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    attr_ind = np.array(attr_ind)
    adj = np.zeros(attr_ind.shape)
    adj[0,:] = attr_ind[1,:]
    adj[1,:] = attr_ind[0,:]
    edge_index = torch.tensor(adj, dtype=torch.long)
    
    H1 = np.expand_dims(np.real(CSI),axis=-1)
    H2 = np.expand_dims(np.imag(CSI),axis=-1)
    HH = np.concatenate((H1,H2),axis=-1)
    y = torch.tensor(np.expand_dims(HH,axis=0), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.contiguous(),edge_attr = edge_attr, y = y)
    return data
def proc_data(HH, dists, norm_csi_real, norm_csi_imag, K):
    n = HH.shape[0]
    data_list = []
    for i in range(n):
        data = build_graph(HH[i,:,:,:],dists[i,:,:], norm_csi_real[i,:,:,:], norm_csi_imag[i,:,:,:], K,500)
        data_list.append(data)
    return data_list

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())#, BN(channels[i])
        for i in range(1, len(channels))
    ])    
class IGConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(IGConv, self).__init__(aggr='max', **kwargs)

        self.mlp1 = mlp1
        self.mlp2 = mlp2
        #self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp1)
        reset(self.mlp2)
        
    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        comb = self.mlp2(tmp)
        nor = torch.sqrt(torch.sum(torch.mul(comb,comb),axis=1))
        nor = nor.unsqueeze(axis=-1)
        comp1 = torch.ones(comb.size(), device=device)
        comb = torch.div(comb,torch.max(comp1,nor) )
        return torch.cat([comb, x[:,:2*Nt]],dim=1)
        
    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1,self.mlp2)
class IGCNet(torch.nn.Module):
    def __init__(self):
        super(IGCNet, self).__init__()

        self.mlp1 = MLP([6*Nt, 64, 64])
        self.mlp2 = MLP([64+4*Nt, 32])
        self.mlp2 = Seq(*[self.mlp2,Seq(Lin(32, 2*Nt))])
        self.conv = IGConv(self.mlp1,self.mlp2)

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x = x0, edge_index = edge_index, edge_attr = edge_attr)
        x2 = self.conv(x = x1, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        return out

def power_check(p):
    n = p.shape[0]
    pp = np.sum(np.square(p),axis=1)
    print(np.sum(pp>1.1))

def sr_loss(data,p,K,N):
    # H1 K*K*N
    # p1 K*N
    H1 = data.y[:,:,:,:,0]
    H2 = data.y[:,:,:,:,1]
    p1 = p[:,:N]
    p2 = p[:,N:2*N]
    p1 = torch.reshape(p1,(-1,K,1,N))
    p2 = torch.reshape(p2,(-1,K,1,N))
    
    rx_power1 = torch.mul(H1, p1)
    rx_power1 = torch.sum(rx_power1,axis=-1)

    rx_power2 = torch.mul(H2, p2)
    rx_power2 = torch.sum(rx_power2,axis=-1)

    rx_power3 = torch.mul(H1, p2)
    rx_power3 = torch.sum(rx_power3,axis=-1)

    rx_power4 = torch.mul(H2, p1)
    rx_power4 = torch.sum(rx_power4,axis=-1)

    rx_power = torch.mul(rx_power1 - rx_power2,rx_power1 - rx_power2) + torch.mul(rx_power3 + rx_power4,rx_power3 + rx_power4)
    mask = torch.eye(K, device = device)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), axis=1)
    interference = torch.sum(torch.mul(rx_power, 1 - mask), axis=1) + 1
    rate = torch.log2(1 + torch.div(valid_rx_power, interference))
    sum_rate = torch.mean(torch.sum(rate, axis=1))
    loss = torch.neg(sum_rate)
    return loss

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = sr_loss(data,out,train_K,Nt)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / train_layouts

def test():
    model.eval()

    total_loss = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            start = time.time()
            out = model(data)
            end = time.time()
            print('CGCNet time:', end-start)
            loss = sr_loss(data,out,test_K,Nt)
            total_loss += loss.item() * data.num_graphs
            #power = out[:,:2*Nt]
            #Y = power.numpy()
            #power_check(Y)
    
    return total_loss / test_layouts

train_K = 30
test_K = 30
train_layouts = 2000
test_layouts = 100
Nt = 2
train_config = init_parameters()
var = 1
train_dists, train_csis = wg.sample_generate(train_config, train_layouts)
test_config = init_parameters()
test_dists, test_csis = wg.sample_generate(test_config, test_layouts)

train_csi_real, train_csi_imag = np.real(train_csis), np.imag(train_csis)
test_csi_real, test_csi_imag = np.real(test_csis), np.imag(test_csis)
norm_train_real, norm_test_real = normalize_data(train_csi_real,test_csi_real, train_config)
norm_train_imag, norm_test_imag = normalize_data(train_csi_imag,test_csi_imag, train_config)

import time
start = time.time()
Y = batch_wmmse(test_csis.transpose(0,2,1,3),var)
end = time.time()
print('WMMSE time:',end-start)
sr = wmmse.IC_sum_rate( test_csis,Y,var)
print('WMMSE rate:',sr)

train_data_list = proc_data(train_csis, train_dists, norm_train_real, norm_train_imag, train_K)
test_data_list = proc_data(test_csis, test_dists, norm_test_real, norm_test_imag,  test_K)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IGCNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)


train_loader = DataLoader(train_data_list, batch_size=64, shuffle=True,num_workers=1)
test_loader = DataLoader(test_data_list, batch_size=test_layouts, shuffle=False, num_workers=1)
for epoch in range(1, 20):
    loss1 = train()
    
    loss2 = test()
    print('Epoch {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(
        epoch, loss1, loss2))
    scheduler.step()


density = test_config.field_length**2/test_K
gen_tests = [40, 80, 160]
for test_K in gen_tests:
    test_layouts = 50
    test_config = init_parameters()
    test_config.n_links = test_K
    test_config.n_receiver = test_K
    field_length = int(np.sqrt(density*test_K))
    test_config.field_length = field_length
    test_dists, test_csis = wg.sample_generate(test_config, test_layouts)
    print('test size', test_csis.shape,field_length)
    
    start = time.time()
    start = time.time()
    Y = batch_wmmse(test_csis.transpose(0,2,1,3),var)
    end = time.time()
    print('WMMSE time:',end-start)
    sr = wmmse.IC_sum_rate( test_csis,Y,var)
    print('WMMSE rate:',sr)

    test_csi_real, test_csi_imag = np.real(test_csis), np.imag(test_csis)
    _, norm_test_real = normalize_data(train_csi_real,test_csi_real, train_config)
    _, norm_test_imag = normalize_data(train_csi_imag,test_csi_imag, test_config)

    test_data_list = proc_data(test_csis, test_dists, norm_test_real, norm_test_imag,  test_K)
    test_loader = DataLoader(test_data_list, batch_size=test_layouts, shuffle=False, num_workers=1)
    loss2 = test()
    print('CGCNet rate:',loss2)