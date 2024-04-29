import torch
import torch.nn as nn
import torch.optim as optim

from dgl.nn import DeepWalk
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data import preprocess
from tqdm import tqdm
import numpy as np
import random
random_seed = 1
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

g,property_labels,house_labels, street_labels, region_labels,out_feats = preprocess("property.csv","house.csv","street.csv","region.csv")
print(g)

model = DeepWalk(g)
dataloader = DataLoader(torch.arange(g.num_nodes()), batch_size=128,
                        shuffle=True, collate_fn=model.sample)
optimizer = SparseAdam(model.parameters(), lr=0.01)
num_epochs = 3

for epoch in tqdm(range(num_epochs)):
    for batch_walk in dataloader:
        loss = model(batch_walk)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_nodes=[i for i in range(14786,28126)]
test_nodes=[i for i in range(28126,33842)]
train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
test_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
train_mask[train_nodes] = True
test_mask[test_nodes] = True
g.ndata['train_mask'] = train_mask
g.ndata['test_mask'] = test_mask

X = model.node_embed.weight.detach()

X=torch.cat((X, out_feats), dim=1).detach()
y = torch.cat((house_labels, property_labels, region_labels, street_labels))
X_train,y_train,X_test,y_test=X[train_mask],y[train_mask],X[test_mask],y[test_mask]

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
input_size = 138  # 输入特征的维度
hidden_size = 50  # 隐藏层大小
output_size = 1  # 输出标签的维度


model = MLP(input_size, hidden_size, output_size)

criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

num_epochs = 200
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs.flatten(), y_train)  # 注意要使用squeeze()去掉输出的维度为1的维度

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# 使用训练好的模型进行预测
with torch.no_grad():
    predicted = model(X_test)
    pre_loss=criterion(predicted.flatten(), y_test)
    print(pre_loss.item())