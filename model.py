import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv, GCNConv, GATConv, ChebConv
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops,dropout_edge

def perturb_features(x, sigma=0.0):
    # Ensure x is a PyTorch tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float)
    
    # Add noise to the tensor
    noise = torch.randn_like(x) * sigma
    return (x + noise).to('cuda')

class MDDGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, intermediate_channels, out_channels, 
                 feature_dims, K=2, attention_hidden=128,perturb_features_p=0.0,dropout_edge_p=0.0):
        super(MDDGCN, self).__init__()
        self.feature_dims = feature_dims
        self.perturb_features_p = perturb_features_p
        self.dropout_edge_p = dropout_edge_p
        self.total_features = sum(feature_dims)
        self.attention_fc = torch.nn.Sequential(
            torch.nn.Linear(self.total_features, attention_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(attention_hidden, len(feature_dims))
        ) 
        self.conv1 = ChebConv(in_channels, hidden_channels, K)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = ChebConv(hidden_channels, intermediate_channels, K)
        self.bn2 = torch.nn.BatchNorm1d(intermediate_channels)
        self.conv3 = ChebConv(intermediate_channels, out_channels, K)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)
        self.fc_main = torch.nn.Sequential(
            torch.nn.Linear(in_channels, intermediate_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(intermediate_channels)
        )
        self.fc_aux = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(out_channels)
        )
        self.alpha_main = torch.nn.Parameter(torch.tensor(0.5)) 
        self.alpha_aux = torch.nn.Parameter(torch.tensor(0.5)) 
        self.dropout = torch.nn.Dropout(p=0.5)

    def apply_dynamic_attention(self, x):
        start = 0
        weighted_x = torch.zeros_like(x)
        raw_weights = self.attention_fc(x.mean(dim=0, keepdim=True)) 
        dynamic_weights = torch.nn.functional.softmax(raw_weights, dim=-1).squeeze(0) 
        for dim, weight in zip(self.feature_dims, dynamic_weights):
            weighted_x[:, start:start + dim] = x[:, start:start + dim] * weight
            start += dim
        return weighted_x

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        initial_x = x.clone() 
        # x = perturb_features(x, sigma=self.perturb_features_p)     
        # edge_index, _ = dropout_edge(edge_index, p=self.dropout_edge_p, force_undirected=True,training=self.training)
        x = self.apply_dynamic_attention(x)
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        initial_x_main = self.fc_main(initial_x) 
        x = F.relu((1 - self.alpha_main) * x + self.alpha_main * initial_x_main)
        x = self.bn3(self.conv3(x, edge_index))
        initial_x_aux = self.fc_aux(initial_x)  
        x = F.relu((1 - self.alpha_aux) * x + self.alpha_aux * initial_x_aux) 
        return x
        
# according to https://github.com/NBStarry/CGMega/blob/main/model.py
class CGMega(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, drop_rate, attn_drop_rate, edge_dim, residual, devices_available):
        super(CGMega, self).__init__()
        self.devices_available = devices_available
        self.drop_rate = drop_rate
        self.convs = torch.nn.ModuleList()
        self.residual = residual
        mid_channels = in_channels + hidden_channels if residual else hidden_channels

        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=heads, dropout=attn_drop_rate, edge_dim=edge_dim,
                                          concat=False, beta=True).to(self.devices_available))
        self.convs.append(TransformerConv(mid_channels, hidden_channels, heads=heads,
                                          dropout=attn_drop_rate, edge_dim=edge_dim, concat=True, beta=True).to(self.devices_available))
        
        self.ln1 = LayerNorm(in_channels=mid_channels).to(self.devices_available)
        self.ln2 = LayerNorm(in_channels=hidden_channels *
                             heads).to(self.devices_available)
        
        self.pool = MaxPool1d(2, 2)

        self.dropout = Dropout(drop_rate)
        self.lins = torch.nn.ModuleList()
        self.lins.append(
            Linear(int(hidden_channels*heads/2), HIDDEN_DIM, weight_initializer="kaiming_uniform").to(devices_available))
        self.lins.append(
            Linear(HIDDEN_DIM, 1, weight_initializer="kaiming_uniform").to(devices_available))
        

    def forward(self, data):
        data = data.to(self.devices_available)
        x = data.x
        edge_index, edge_attr = dropout_adj(data.edge_index, data.edge_attr, p=self.drop_rate, force_undirected=True,
                                            training=self.training)
        res = x
        x = self.convs[0](x, edge_index, edge_attr)
        x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE, inplace=True)
        x = torch.cat((x, res), dim=1) if self.residual else x
        x = self.ln1(x)

        edge_index, edge_attr = dropout_adj(data.edge_index, data.edge_attr, p=self.drop_rate, force_undirected=True,
                                            training=self.training)
        x = self.convs[1](x.to(self.devices_available), edge_index.to(
            self.devices_available), edge_attr.to(self.devices_available))
        x = self.ln2(x)
        x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE)
        x = torch.unsqueeze(x, 1)
        x = self.pool(x)
        x = torch.squeeze(x)
        x = self.lins[0](x).relu()
        x = self.dropout(x)
        x = self.lins[1](x)

        return torch.sigmoid(x)

# according to https://github.com/Bibyutatsu/proEMOGI/blob/main/proEMOGI/proemogi.py
class EMOGI(torch.nn.Module):
    def __init__(self, in_channels, devices_available, num_hidden_layers=2, drop_rate=0.5,
                 hidden_dims=[20, 40], pos_loss_multiplier=1, weight_decay=5e-4,):
        super(EMOGI, self).__init__()
        self.in_channels = in_channels
        self.devices_available = devices_available

        # model params
        self.weight_decay = weight_decay
        self.pos_loss_multiplier = pos_loss_multiplier
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dims = hidden_dims
        self.drop_rate = drop_rate
        
        self.convs = torch.nn.ModuleList()

        # add intermediate layers
        inp_dim = self.in_channels
        for l in range(self.num_hidden_layers):
            self.convs.append(GCNConv(inp_dim,
                                       self.hidden_dims[l]).to(self.devices_available))
            inp_dim = self.hidden_dims[l]
            
        self.convs.append(GCNConv(self.hidden_dims[-1], 1).to(self.devices_available))
        
    def forward(self, data):
        data = data.to(self.devices_available)
        x = data.x
        for layer in self.convs[:-1]:
            x = layer(x, data.edge_index)
            x = F.relu(x)
            if self.drop_rate is not None:
                x = F.dropout(x, self.drop_rate, training=self.training)
        x = self.convs[-1](x, data.edge_index)
        return torch.sigmoid(x)
    