import scipy.io as sio                     
import numpy as np                         
import matplotlib.pyplot as plt           
import function_wmmse_powercontrol as wf
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
import wireless_networks_generator as wg
from FPLinQ import FP_optimize, FP
import helper_functions
import time

class init_parameters():
    def __init__(self):
        # wireless network settings
        self.n_links = train_K
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
        self.tx_power = np.power(10, (self.tx_power_milli_decibel-30)/10)
        self.noise_density_milli_decibel = -169
        self.input_noise_power = np.power(10, ((self.noise_density_milli_decibel-30)/10)) * self.bandwidth
        self.output_noise_power = self.input_noise_power
        self.SNR_gap_dB = 6
        self.SNR_gap = np.power(10, self.SNR_gap_dB/10)
        self.setting_str = "{}_links_{}X{}_{}_{}_length".format(self.n_links, self.field_length, self.field_length, self.shortest_directLink_length, self.longest_directLink_length)
        # 2D occupancy grid setting
        self.cell_length = 5
        self.n_grids = np.round(self.field_length/self.cell_length).astype(int)

def proc_train_losses(train_path_losses, train_channel_losses):
    mask = np.eye(train_K)
    diag_path = np.multiply(mask,train_path_losses)
    off_diag_path = train_path_losses - diag_path
    diag_channel = np.multiply(mask,train_channel_losses)
    train_losses = diag_channel + off_diag_path
    return train_losses

def normalize_data(train_data,test_data):
    mask = np.eye(train_K)
    train_copy = np.copy(train_data)
    diag_H = np.multiply(mask,train_copy)
    diag_mean = np.sum(diag_H)/train_layouts/train_K
    diag_var = np.sqrt(np.sum(np.square(diag_H))/train_layouts/train_K)
    tmp_diag = (diag_H - diag_mean)/diag_var

    off_diag = train_copy - diag_H
    off_diag_mean = np.sum(off_diag)/train_layouts/train_K/(train_K-1)
    off_diag_var = np.sqrt(np.sum(np.square(off_diag))/train_layouts/train_K/(train_K-1))
    tmp_off = (off_diag - off_diag_mean)/off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off,mask)
    
    norm_train = np.multiply(tmp_diag,mask) + tmp_off_diag
    
    # normlize test
    mask = np.eye(test_K)
    test_copy = np.copy(test_data)
    diag_H = np.multiply(mask,test_copy)
    tmp_diag = (diag_H - diag_mean)/diag_var
    
    off_diag = test_copy - diag_H
    tmp_off = (off_diag - off_diag_mean)/off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off,mask)
    
    norm_test = np.multiply(tmp_diag,mask) + tmp_off_diag
    
    return norm_train, norm_test

def simple_greedy(X,AAA,label):
    
    n = X.shape[0]
    thd = int(np.sum(label)/n)
    Y = np.zeros((n,test_K))
    for ii in range(n):
        alpha = AAA[ii,:]
        H_diag = alpha * np.square(np.diag(X[ii,:,:]))
        xx = np.argsort(H_diag)[::-1]
        for jj in range(thd):
            Y[ii,xx[jj]] = 1
    return Y

def build_graph(loss, dist, norm_dist, norm_loss, K, threshold):
    n = loss.shape[0]
    x1 = np.expand_dims(np.diag(norm_dist),axis=1)
    x2 = np.expand_dims(np.diag(norm_loss),axis=1)
    x3 = np.zeros((K,1))
    x = np.concatenate((x1,x2,x3),axis=1)
    x = torch.tensor(x, dtype=torch.float)
    
    dist2 = np.copy(dist)
    mask = np.eye(K)
    diag_dist = np.multiply(mask,dist2)
    dist2 = dist2 + 1000* diag_dist
    dist2[dist2 > threshold] = 0
    attr_ind = np.nonzero(dist2)
    edge_attr = norm_dist[attr_ind]
    edge_attr = np.expand_dims(edge_attr, axis = -1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    attr_ind = np.array(attr_ind)
    adj = np.zeros(attr_ind.shape)
    adj[0,:] = attr_ind[1,:]
    adj[1,:] = attr_ind[0,:]
    edge_index = torch.tensor(adj, dtype=torch.long)

    y = torch.tensor(np.expand_dims(loss,axis=0), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.contiguous(),edge_attr = edge_attr, y = y)
    return data

def proc_data(HH, dists, norm_dists, norm_HH, K):
    n = HH.shape[0]
    data_list = []
    for i in range(n):
        data = build_graph(HH[i,:,:],dists[i,:,:], norm_dists[i,:,:], norm_HH[i,:,:], K,300)
        data_list.append(data)
    return data_list

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
        return torch.cat([x[:,:2], comb],dim=1)
        
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

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())#, BN(channels[i]))
        for i in range(1, len(channels))
    ])
class IGCNet(torch.nn.Module):
    def __init__(self):
        super(IGCNet, self).__init__()

        self.mlp1 = MLP([4, 32, 32])
        self.mlp2 = MLP([35, 16])
        self.mlp2 = Seq(*[self.mlp2,Seq(Lin(16, 1), Sigmoid())])
        self.conv = IGConv(self.mlp1,self.mlp2)

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x = x0, edge_index = edge_index, edge_attr = edge_attr)
        x2 = self.conv(x = x1, edge_index = edge_index, edge_attr = edge_attr)
        #x3 = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        #x4 = self.conv(x = x3, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        return out
        

def sr_loss(data, out, K):
    power = out[:,2]
    power = torch.reshape(power, (-1, K, 1)) 
    
    abs_H_2 = data.y#torch.pow(abs_H, 2)
    abs_H_2 = abs_H_2.permute(0,2,1)
    rx_power = torch.mul(abs_H_2, power)
    mask = torch.eye(K)
    mask = mask.to(device)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)
    interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + var
    rate = torch.log2(1 + torch.div(valid_rx_power, interference))
    #w_rate = torch.mul(data.pos,rate)
    sum_rate = torch.mean(torch.sum(rate, 1))
    loss = torch.neg(sum_rate)
    return loss

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = sr_loss(data,out,train_K)
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
            print('dnn time',end-start)
            loss = sr_loss(data,out,test_K)
            total_loss += loss.item() * data.num_graphs
    power = out[:,2]
    power = torch.reshape(power, (-1, test_K)) 
    Y = power.numpy()
    rates = helper_functions.compute_rates(test_config, 
            Y, directLink_channel_losses, crossLink_channel_losses)
    sr = np.mean(np.sum(rates,axis=1))
    print('actual_rates:',sr)
    
    return total_loss / test_layouts




train_K = 50
test_K = 50
train_layouts = 2000
test_layouts = 500

train_config = init_parameters()
var = train_config.output_noise_power / train_config.tx_power
layouts, train_dists = wg.generate_layouts(train_config, train_layouts)
train_path_losses = wg.compute_path_losses(train_config, train_dists)
train_channel_losses = helper_functions.add_shadowing(train_path_losses)
train_channel_losses = helper_functions.add_fast_fading(train_channel_losses)
train_losses = proc_train_losses(train_path_losses, train_channel_losses)

test_config = init_parameters()
layouts, test_dists = wg.generate_layouts(test_config, test_layouts)
test_path_losses = wg.compute_path_losses(test_config, test_dists)
test_channel_losses = helper_functions.add_shadowing(test_path_losses)
test_channel_losses = helper_functions.add_fast_fading(test_channel_losses)

norm_train_dists, norm_test_dists = normalize_data(1/train_dists,1/test_dists)
norm_train_losses, norm_test_losses = normalize_data(np.sqrt(train_channel_losses),np.sqrt(test_channel_losses) )

train_data_list = proc_data(train_losses, train_dists, norm_train_dists, norm_train_losses, train_K)
test_data_list = proc_data(test_channel_losses, test_dists, norm_test_dists, norm_test_losses, test_K)


directLink_channel_losses = helper_functions.get_directLink_channel_losses(test_channel_losses)
crossLink_channel_losses = helper_functions.get_crossLink_channel_losses(test_channel_losses)

Pini = np.random.rand(test_layouts,test_K,1 )
Y1 = FP_optimize(test_config,test_channel_losses,np.ones([test_layouts, test_K]))
Y2 = wf.batch_WMMSE2(Pini,np.ones([test_layouts, test_K]),np.sqrt(test_channel_losses),1,var)


rates1 = helper_functions.compute_rates(test_config, 
            Y1, directLink_channel_losses, crossLink_channel_losses)
rates2 = helper_functions.compute_rates(test_config, 
            Y2, directLink_channel_losses, crossLink_channel_losses)


sr1 = np.mean(np.sum(rates1,axis=1))
sr2 = np.mean(np.sum(rates2,axis=1))

print('FPlinQ fade:',sr1)
print('WMMSE fade:',sr2)

bl_Y = simple_greedy(test_channel_losses,np.ones([test_layouts, test_K]),Y1)

rates_bl = helper_functions.compute_rates(test_config, 
            bl_Y, directLink_channel_losses, crossLink_channel_losses)
sr_bl = np.mean(np.sum(rates_bl,axis=1))
print('baseline:',sr_bl)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IGCNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
train_loader = DataLoader(train_data_list, batch_size=64, shuffle=True,num_workers=1)
test_loader = DataLoader(test_data_list, batch_size=test_layouts, shuffle=False, num_workers=1)
for epoch in range(1, 50):
    loss1 = train()
    if(epoch % 4 == 0):
        loss2 = test()
        print('Epoch {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(
            epoch, loss1, loss2))
    scheduler.step()


density = test_config.field_length**2/test_K
gen_tests = [20, 50, 100, 200, 300, 400, 500]
for test_K in gen_tests:
    test_layouts = 50
    test_config = init_parameters()
    test_config.n_links = test_K
    field_length = int(np.sqrt(density*test_K))
    test_config.field_length = field_length
    layouts, test_dists = wg.generate_layouts(test_config, test_layouts)
    test_path_losses = wg.compute_path_losses(test_config, test_dists)
    test_channel_losses = helper_functions.add_shadowing(test_path_losses)
    test_channel_losses = helper_functions.add_fast_fading(test_channel_losses)

    print('test size', layouts.shape,field_length)
    directLink_channel_losses = helper_functions.get_directLink_channel_losses(test_channel_losses)
    crossLink_channel_losses = helper_functions.get_crossLink_channel_losses(test_channel_losses)
    Y = FP_optimize(test_config,test_channel_losses,np.ones([test_layouts, test_K]))

    Pini = np.random.rand(test_layouts,test_K,1 )
    Y2 = wf.batch_WMMSE2(Pini,np.ones([test_layouts, test_K]),np.sqrt(test_channel_losses),1,var)
    rates1 = helper_functions.compute_rates(test_config, 
                Y, directLink_channel_losses, crossLink_channel_losses)
    rates2 = helper_functions.compute_rates(test_config, 
                Y2, directLink_channel_losses, crossLink_channel_losses)
    sr1 = np.mean(np.sum(rates1,axis=1))
    sr2 = np.mean(np.sum(rates2,axis=1))
    print('FPlinQ:',sr1)
    print('WMMSE:',sr2)

    bl_Y = simple_greedy(test_channel_losses,np.ones([test_layouts, test_K]),Y)

    rates_bl = helper_functions.compute_rates(test_config, bl_Y, directLink_channel_losses, crossLink_channel_losses)
    sr_bl = np.mean(np.sum(rates_bl,axis=1))
    print('baseline:',sr_bl)
    _, norm_test_dists = normalize_data(1/train_dists,1/test_dists)
    norm_train_losses, norm_test_losses = normalize_data(np.sqrt(train_channel_losses),np.sqrt(test_channel_losses) )
    test_data_list = proc_data(test_channel_losses, test_dists, norm_test_dists, norm_test_losses, test_K)
    test_loader = DataLoader(test_data_list, batch_size=test_layouts, shuffle=False, num_workers=1)
    loss2 = test()
    print('CGCNet:',loss2)




