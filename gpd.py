import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        return F.normalize(F.relu(aggr_out), p=2, dim=-1)


class GatedNetwork(nn.Module):
    def __init__(self, node_channels, edge_channels):
        super(GatedNetwork, self).__init__()
        self.A = nn.Linear(node_channels, edge_channels)
        self.B = nn.Linear(node_channels, edge_channels)
        self.C = nn.Linear(edge_channels, edge_channels)
        self.D = nn.Linear(edge_channels, edge_channels)
        self.bn = nn.BatchNorm1d(edge_channels)
        self.U = nn.Linear(node_channels, node_channels)
        self.V = nn.Linear(node_channels, node_channels)
        self.epsilon = 1e-5

    def forward(self, h, edge_index, e):
        row, col = edge_index
        e_out = self.A(h[row]) + self.B(h[col]) + \
            self.C(e[col]) + self.D(e[row])
        e_out = F.relu(self.bn(e_out))

        e_new = e.clone()
        e_new[row] += e_out
        e_new = F.sigmoid(e_new)
        e_new = e_new / (torch.sum(e_new, dim=0, keepdim=True) + self.epsilon)

        h_out = self.U(h) + (e_new * self.V(h[col])).sum(dim=0)
        h_out = F.relu(h_out)

        return h_out, e_new

class GraphClassificationNetwork(nn.Module):
    def __init__(self, in_channels, hc1, hc2, hc3, hc4, num_classes):
        super(GraphClassificationNetwork, self).__init__()
        self.gcn1 = GCNConv(in_channels, hc1)
        self.gcn2 = GCNConv(hc1, hc2)
        self.gated1 = GatedNetwork(hc2, hc3)
        self.gated2 = GatedNetwork(hc3, hc4)
        self.classifier = nn.Linear(hc4, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Two-layer GCN, initial node embedding 
        h = self.gcn1(x, edge_index, edge_attr)
        h = self.gcn2(h, edge_index, edge_attr)

        # Initial edge embedding by concatenating the node embeddings
        e = torch.cat((h[edge_index[0]], h[edge_index[1]]), dim=1)

        # Two-layer gated network, updated edge embedding
        h, e = self.gated1(h, edge_index, e)
        h, e = self.gated2(h, edge_index, e)

        # Graph pooling: mean of the node embeddings
        graph_embedding = torch.mean(h, dim=0)

        # Classification
        out = self.classifier(graph_embedding)
        return F.log_softmax(out, dim=-1)



def train(model, data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out.unsqueeze(0), data.y)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        pred = logits.argmax(dim=-1)
        correct = (pred == data.y)
    return correct.item()

data = None # placeholder

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model and optimizer
in_channels = data.num_features
num_classes = 3  # Three categories
model = GraphClassificationNetwork(in_channels, hc1=64, hc2=32, hc3=128, hc4=64, num_classes=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = F.nll_loss