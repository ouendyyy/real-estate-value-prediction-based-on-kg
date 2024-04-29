
import numpy as np

import matplotlib.pyplot as plt
from model import *
import random
from torch.nn.functional import normalize
from tqdm import tqdm
from data import preprocess
import torch.nn.functional as F
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

G,feats,labels=preprocess("property.csv","house.csv","street.csv","region.csv")
#labels 是tensor
print(G)
print(labels)
device = torch.device("cuda:0")
G.node_dict = {}
G.edge_dict = {}

for ntype in G.ntypes:

    G.node_dict[ntype] = len(G.node_dict)
for etype in G.etypes:
    G.edge_dict[etype] = len(G.edge_dict)
    G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * G.edge_dict[etype]

G = G.to(device)

pid=np.array([i for i in range(len(labels))])
shuffle = np.random.permutation(pid)
train_idx = torch.tensor(shuffle[0:12340]).long()
val_idx = torch.tensor(shuffle[12340:17500]).long()
test_idx = torch.tensor(shuffle[17500:]).long()

for ntype in G.ntypes:
    feats[ntype]=normalize(feats[ntype],p=2,dim=1)
    G.nodes[ntype].data['inp'] = feats[ntype].to(device)

model = HGT(G, n_inp=70, n_hid=256, n_out=1, n_layers=2, n_heads=4, use_norm=True).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=200, max_lr=1e-3, pct_start=0.05)

train_losses = []            #保存loss
criterion = nn.MSELoss()
best_val_rmse = float('inf')
best_test_rmse = float('inf')
train_step = 0
for epoch in tqdm(range(200)):
    logits = model(G, 'property')
    # The loss is computed only for labeled nodes.
    loss = criterion(logits[train_idx].flatten(), labels[train_idx].to(device))
    pred = logits.cpu()
    train_rmse = loss
    val_rmse = criterion(pred[val_idx].flatten(), labels[val_idx])
    test_rmse = criterion(pred[test_idx].flatten(), labels[test_idx])

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    train_step += 1
    scheduler.step(train_step)
    train_losses.append(train_rmse.item())
    if best_val_rmse > val_rmse:
        best_val_rmse = val_rmse
        best_test_rmse = test_rmse

    if epoch % 5 == 0:
        print('LR: %.5f Loss %.4f, Train MSE %.4f, Test MSE %.4f (Best %.4f)' % (
            optimizer.param_groups[0]['lr'],
            loss.item(),
            train_rmse.item(),
            test_rmse.item(),
            best_test_rmse,
        ))


plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss Curve')
plt.legend()
plt.show()