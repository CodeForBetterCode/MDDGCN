import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv, GCNConv, GATConv
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
                 feature_dims, attention_hidden=128, perturb_features_p=0.0, dropout_edge_p=0.0):
        super().__init__()
        self.feature_dims = feature_dims
        self.total_features = sum(feature_dims)
        self.perturb_features_p = perturb_features_p
        self.attention_fc = torch.nn.Sequential(
            torch.nn.Linear(self.total_features, attention_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(attention_hidden, len(feature_dims))
        )

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, intermediate_channels)
        self.bn2 = torch.nn.BatchNorm1d(intermediate_channels)
        self.conv3 = GCNConv(intermediate_channels, out_channels)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if in_channels == hidden_channels:
            self.res_connector1 = torch.nn.Identity()
        else:
            self.res_connector1 = torch.nn.Linear(in_channels, hidden_channels)
            
        if hidden_channels == intermediate_channels:
            self.res_connector2 = torch.nn.Identity()
        else:
            self.res_connector2 = torch.nn.Linear(hidden_channels, intermediate_channels)

        self.fc_main = torch.nn.Linear(in_channels, intermediate_channels)
        self.fc_aux = torch.nn.Linear(in_channels, out_channels)

        self.gate_main = torch.nn.Sequential(torch.nn.Linear(intermediate_channels * 2, 1), torch.nn.Sigmoid())
        self.gate_aux = torch.nn.Sequential(torch.nn.Linear(out_channels * 2, 1), torch.nn.Sigmoid())
        
        self.dropout = torch.nn.Dropout(0.4)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(out_channels, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def apply_node_attention(self, x):
        start = 0
        weights = F.softmax(self.attention_fc(x), dim=-1)
        out = torch.zeros_like(x)
        for i, dim in enumerate(self.feature_dims):
            out[:, start:start+dim] = x[:, start:start+dim] * weights[:, i].unsqueeze(-1)
            start += dim
        return out

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        initial_x = x
        x = perturb_features(x, sigma=self.perturb_features_p)     
        x_after_attention = self.apply_node_attention(x)
        x1_out = F.relu(self.bn1(self.conv1(x_after_attention, edge_index)))
        x1_out = self.dropout(x1_out)
        res1 = self.res_connector1(x_after_attention)
        x_after_conv1 = x1_out + res1
        x2_out = F.relu(self.bn2(self.conv2(x_after_conv1, edge_index)))
        x_after_conv2 = self.dropout(x2_out)
        res2 = self.res_connector2(x_after_conv1)
        x = x_after_conv2 + res2
        main_res = F.relu(self.fc_main(initial_x))
        gate_m_input = torch.cat([x, main_res], dim=-1)
        gate_m = self.gate_main(gate_m_input)
        x = gate_m * x + (1 - gate_m) * main_res
        x = self.bn3(self.conv3(x, edge_index))
        aux_res = F.relu(self.fc_aux(initial_x))
        gate_a_input = torch.cat([x, aux_res], dim=-1)
        gate_a = self.gate_aux(gate_a_input)
        x = gate_a * x + (1 - gate_a) * aux_res
        return self.classifier(x)

        
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
    
