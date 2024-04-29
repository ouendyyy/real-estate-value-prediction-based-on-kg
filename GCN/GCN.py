import torch

from data import preprocess
import torch.nn as nn
import numpy as np
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
import random
random_seed = 1
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

G,property_labels,house_labels, street_labels, region_labels = preprocess("property.csv","house.csv","street.csv","region.csv")
print(G)
feats=G.ndata['hv']
print(feats.shape)
labels = torch.cat((house_labels, property_labels, region_labels, street_labels))

train_nodes=[i for i in range(14786,28126)]
test_nodes=[i for i in range(28126,33842)]
train_mask = torch.zeros(G.num_nodes(), dtype=torch.bool)
test_mask = torch.zeros(G.num_nodes(), dtype=torch.bool)
train_mask[train_nodes] = True
test_mask[test_nodes] = True
G.ndata['train_mask'] = train_mask
G.ndata['test_mask'] = test_mask

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_size)
        self.conv2 = dglnn.GraphConv(hidden_size, out_feats)

    def forward(self, g, features):
        # 第一层GCN
        h = F.relu(self.conv1(g, features))
        # 第二层GCN
        h = self.conv2(g, h)
        return h

model = GCN(in_feats=20, hidden_size=128, out_feats=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 200
for epoch in range(num_epochs):
    # 前向传播
    logits = model(G, feats)
    # 计算损失
    loss = F.mse_loss(logits[train_mask].squeeze(), labels[train_mask])
    # 反向传播
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    # 打印损失
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

with torch.no_grad():
    logits = model(G,feats)
    pre=logits[test_mask].squeeze()
    lab=labels[test_mask]
    test_loss = F.mse_loss(pre.squeeze(), labels[test_mask])
    print(f"MSE Loss: {test_loss.item()}")