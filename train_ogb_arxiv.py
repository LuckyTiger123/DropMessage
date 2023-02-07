import os
import sys
import torch
import argparse
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from src.layer import GNNLayer

# parse parameters
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda-device', type=int, default=0, help='which gpu device to use.')
parser.add_argument('-dr', '--dropping-rate', type=float, default=0, help='The chosen dropping rate (default: 0).')
parser.add_argument('-e', '--epoch', type=int, default=500, help='The epoch number (default: 500).')
parser.add_argument('-bb', '--backbone', type=str, default='GCN', help='The backbone model [GCN, GAT, APPNP].')
parser.add_argument('-dm', '--dropping-method', type=str, default='DropMessage',
                    help='The chosen dropping method [Dropout, DropEdge, DropNode, DropMessage].')
parser.add_argument('-k', '--K', type=int, default=10, help='The K value for APPNP (default: 10).')
parser.add_argument('-a', '--alpha', type=float, default=0.1, help='The alpha value for APPNP (default: 0.1).')
parser.add_argument('-hd', '--hidden-dimension', type=int, default=128,
                    help='The hidden dimension number (default: 128).')
parser.add_argument('-nl', '--num-layers', type=int, default=3, help='The layer number (default: 3)')
parser.add_argument('-r', '--rand-seed', type=int, default=0, help='The random seed (default: 0).')
args = parser.parse_args()

train_dataset = 'ogbn-arxiv'
epoch_num = args.epoch

# random seed setting
random_seed = args.rand_seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# device selection
device = torch.device('cuda:{}'.format(args.cuda_device) if torch.cuda.is_available() else 'cpu')

# load dataset
dataset = PygNodePropPredDataset(name=train_dataset, root='/home/luckytiger/TestDataset', transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)
split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)
evaluator = Evaluator(name=train_dataset)


# Model
class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, backbone, dropping_method):
        super(Model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GNNLayer(in_channels, hidden_channels, dropping_method, backbone, alpha=args.alpha, K=args.K))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GNNLayer(hidden_channels, hidden_channels, dropping_method, backbone, alpha=args.alpha, K=args.K))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GNNLayer(hidden_channels, out_channels, dropping_method, backbone, alpha=args.alpha, K=args.K))

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, drop_rate)
            x = self.bns[i](x)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index, drop_rate)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t, args.dropping_rate)
    loss = F.cross_entropy(out[train_idx], data.y.squeeze(1)[train_idx])
    loss.backward()
    print('the train loss is {}'.format(float(loss)))
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


model = Model(data.num_features, args.hidden_dimension, dataset.num_classes, args.num_layers, args.backbone,
              args.dropping_method).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

model.reset_parameters()
best_val = best_test = 0
for epoch in range(epoch_num):
    train()
    train_acc, val_acc, current_test_acc = test()
    print('---------------------------------------------------------------------------')
    print('For the {} epoch, the train acc is {}, '
          'the val acc is {}, the test acc is {}.'.format(epoch,
                                                          train_acc,
                                                          val_acc,
                                                          current_test_acc))
    if val_acc > best_val:
        best_val = val_acc
        best_test = current_test_acc

print('Mission completes.')

print('--------------------------------------------------------------------------')

print('Dataset: {}.'.format(train_dataset))
print('Backbone model: {}. Dropping method: {}.'.format(args.backbone, args.dropping_method))
print('The final test acc is {}.'.format(best_test))
