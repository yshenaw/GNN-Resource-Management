import scipy.io as sio                     # import scipy.io for .mat file I/O 
import numpy as np                         # import numpy
import matplotlib.pyplot as plt            # import matplotlib.pyplot for figure plotting
import function_wmmse_powercontrol as wf
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN
import time
def generate_wGaussian(K, num_H, var_noise=1, Pmin=0, seed=2017):
    print('Generate Data ... (seed = %d)' % seed)
    np.random.seed(seed)
    Pmax = 1
    Pini = Pmax*np.ones((num_H,K,1) )
    #alpha = np.random.rand(num_H,K)
    alpha = np.random.rand(num_H,K)
    #alpha = np.ones((num_H,K))
    fake_a = np.ones((num_H,K))
    #var_noise = 1
    X=np.zeros((K**2,num_H))
    Y=np.zeros((K,num_H))
    total_time = 0.0
    CH = 1/np.sqrt(2)*(np.random.randn(num_H,K,K)+1j*np.random.randn(num_H,K,K))
    H=abs(CH)
    Y = wf.batch_WMMSE2(Pini,alpha,H,Pmax,var_noise)
    Y2 = wf.batch_WMMSE2(Pini,fake_a,H,Pmax,var_noise)
    return H, Y, alpha, Y2

def np_sum_rate(H,p,alpha,var_noise):
    H = np.expand_dims(H,axis=-1)
    K = H.shape[1]
    N = H.shape[-1]
    p = p.reshape((-1,K,1,N))
    rx_power = np.multiply(H, p)
    rx_power = np.sum(rx_power,axis=-1)
    rx_power = np.square(abs(rx_power))
    mask = np.eye(K)
    valid_rx_power = np.sum(np.multiply(rx_power, mask), axis=1)
    interference = np.sum(np.multiply(rx_power, 1 - mask), axis=1) + var_noise
    rate = np.log(1 + np.divide(valid_rx_power, interference))
    w_rate = np.multiply(alpha,rate)
    sum_rate = np.mean(np.sum(w_rate, axis=1))
    return sum_rate

def get_cg(n):
    adj = []
    for i in range(0,n):
        for j in range(0,n):
            if(not(i==j)):
                adj.append([i,j])
    return adj

def build_graph(H,A,adj):
    n = H.shape[0]
    x1 = np.expand_dims(np.diag(H),axis=1)
    x2 = np.expand_dims(A,axis=1)
    x3 = np.ones((K,1))
    edge_attr = []
    
    x = np.concatenate((x1,x2,x3),axis=1)
    for e in adj:
        edge_attr.append([H[e[0],e[1]],H[e[1],e[0]]])
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(adj, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor(np.expand_dims(H,axis=0), dtype=torch.float)
    pos = torch.tensor(np.expand_dims(A,axis=0), dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index.t().contiguous(),edge_attr = edge_attr, y = y, pos = pos)
    return data 

def proc_data(HH,AA):
    n = HH.shape[0]
    data_list = []
    cg = get_cg(K)
    for i in range(n):
        data = build_graph(HH[i],AA[i],cg)
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
        Seq(Lin(channels[i - 1], channels[i], bias = True), ReLU())#, BN(channels[i]))
        for i in range(1, len(channels))
    ])
class IGCNet(torch.nn.Module):
    def __init__(self):
        super(IGCNet, self).__init__()

        self.mlp1 = MLP([5, 16, 32])
        self.mlp2 = MLP([35, 16])
        self.mlp2 = Seq(*[self.mlp2,Seq(Lin(16, 1, bias = True), Sigmoid())])
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
    abs_H = data.y
    abs_H_2 = torch.pow(abs_H, 2)  
    rx_power = torch.mul(abs_H_2, power)
    mask = torch.eye(K)
    mask = mask.to(device)
    valid_rx_power = torch.sum(torch.mul(rx_power, mask), 1)
    interference = torch.sum(torch.mul(rx_power, 1 - mask), 1) + var
    rate = torch.log(1 + torch.div(valid_rx_power, interference))
    w_rate = torch.mul(data.pos,rate)
    sum_rate = torch.mean(torch.sum(w_rate, 1))
    loss = torch.neg(sum_rate)
    return loss

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = sr_loss(data,out,K)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / num_H

def test():
    model.eval()

    total_loss = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            loss = sr_loss(data,out,K)
            total_loss += loss.item() * data.num_graphs
    return total_loss / num_test

def simple_greedy(X,AAA,label):
    
    n = X.shape[0]
    thd = int(np.sum(label)/n)
    Y = np.zeros((n,K))
    for ii in range(n):
        alpha = AAA[ii,:]
        H_diag = alpha * np.square(np.diag(X[ii,:,:]))
        xx = np.argsort(H_diag)[::-1]
        for jj in range(thd):
            Y[ii,xx[jj]] = 1
    return Y


K = 10              # number of users
num_H = 10000          # number of training samples
num_test = 2000            # number of testing  samples
training_epochs = 50      # number of training epochs
trainseed = 0              # set random seed for training set
testseed = 7               # set random seed for test set
print('Gaussian IC Case: K=%d, Total Samples: %d, Total Iterations: %d\n'%(K, num_H, training_epochs))
var_db = 10
var = 1/10**(var_db/10)
Xtrain, Ytrain, Atrain, wtime = generate_wGaussian(K, num_H, seed=trainseed, var_noise = var)
X, Y, A, Y2 = generate_wGaussian(K, num_test, seed=testseed, var_noise = var)
bl_Y = simple_greedy(X,A,Y)
print('greedy:',np_sum_rate(X,bl_Y,A,var))

print('wmmse:',np_sum_rate(X.transpose(0,2,1),Y,A,var))
print('wmmse unweighted:',np_sum_rate(X.transpose(0,2,1),Y2,A,var))
train_data_list = proc_data(Xtrain,Atrain)
test_data_list = proc_data(X,A)   
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
model = IGCNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
train_loader = DataLoader(train_data_list, batch_size=64, shuffle=True,num_workers=1)
test_loader = DataLoader(test_data_list, batch_size=2000, shuffle=False, num_workers=1)
for epoch in range(1, 200):
    loss1 = train()
    if(epoch % 8 == 0):
        loss2 = test()
        print('Epoch {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(
            epoch, loss1, loss2))
    scheduler.step()

