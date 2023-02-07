import os
import sys
import torch
import argparse
from torch import Tensor
from torch_geometric.datasets import Flickr
from torch_geometric.typing import Adj
import torch.nn.functional as F

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
parser.add_argument('-hs', '--heads', type=int, default=1, help='The head number for GAT (default: 1).')
parser.add_argument('-k', '--K', type=int, default=10, help='The K value for APPNP (default: 10).')
parser.add_argument('-a', '--alpha', type=float, default=0.1, help='The alpha value for APPNP (default: 0.1).')
parser.add_argument('-fyd', '--first-layer-dimension', type=int, default=256,
                    help='The hidden dimension number for two-layer GNNs (default: 256).')
parser.add_argument('-r', '--rand-seed', type=int, default=0, help='The random seed (default: 0).')
args = parser.parse_args()

# random seed setting
random_seed = args.rand_seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# device selection
device = torch.device('cuda:{}'.format(args.cuda_device) if torch.cuda.is_available() else 'cpu')

# load dataset
dataset = Flickr(root='/home/luckytiger/TestDataset/Flickr')
data = dataset[0].to(device)


# Model
class Model(torch.nn.Module):
    def __init__(self, feature_num, output_num, backbone, dropping_method):
        super(Model, self).__init__()
        self.backbone = backbone
        self.gnn1 = GNNLayer(feature_num, args.first_layer_dimension, dropping_method, backbone, heads=args.heads,
                             alpha=args.alpha, K=args.K)
        self.gnn2 = GNNLayer(args.first_layer_dimension * args.heads, output_num, dropping_method, backbone,
                             alpha=args.alpha, K=args.K)

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        x = self.gnn1(x, edge_index, drop_rate)
        if self.backbone == 'GAT':
            x = F.elu(x)
        else:
            x = F.relu(x)
        x = self.gnn2(x, edge_index, drop_rate)
        return x

    def reset_parameters(self):
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()


model = Model(dataset.num_features, dataset.num_classes, args.backbone, args.dropping_method).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
epoch_num = args.epoch


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, args.dropping_rate)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    print('the train loss is {}'.format(float(loss)))
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index, args.dropping_rate)
    _, pred = out.max(dim=1)
    train_correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    train_acc = train_correct / int(data.train_mask.sum())
    validate_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
    validate_acc = validate_correct / int(data.val_mask.sum())
    test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    test_acc = test_correct / int(data.test_mask.sum())
    return train_acc, validate_acc, test_acc


best_val_acc = test_acc = 0
for epoch in range(epoch_num):
    train()
    train_acc, val_acc, current_test_acc = test()
    print('For the {} epoch, the train acc is {}, the val acc is {}, the test acc is {}.'.format(epoch, train_acc,
                                                                                                 val_acc,
                                                                                                 current_test_acc))
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = current_test_acc

print('Mission completes.')

print('--------------------------------------------------------------------------')

print('Dataset: {}.'.format('Flickr'))
print('Backbone model: {}. Dropping method: {}.'.format(args.backbone, args.dropping_method))
print('The final test acc is {}.'.format(test_acc))
