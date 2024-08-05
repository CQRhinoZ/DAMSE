import math
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GraphConv, GraphNorm
from torch_geometric.loader import DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Parameter, TransformerDecoder, TransformerDecoderLayer


class GcnUnit(nn.Module):
    def __init__(self, gcn_channel=128):
        super(GcnUnit, self).__init__()
        self.cov1 = GCNConv(gcn_channel, gcn_channel)
        self.normal = GraphNorm(gcn_channel)
        self.cov2 = GATConv(gcn_channel, gcn_channel)
        self.relu = nn.LeakyReLU()

    def forward(self, x, edges):
        temp = self.cov1(x, edges)
        temp = self.normal(temp)
        temp = self.relu(temp)
        x = x + temp
        temp = self.cov2(x, edges)
        temp = self.normal(temp)
        temp = self.relu(temp)
        x = x + temp
        return x


class AGCNUnit(nn.Module):
    def __init__(self, gcn_channel=128):
        super(AGCNUnit, self).__init__()
        self.cov = GATConv(gcn_channel, gcn_channel)
        self.cov = GATConv(gcn_channel, gcn_channel)
        self.relu = nn.LeakyReLU()

    def forward(self, x, edges):
        temp = self.cov(x, edges)
        x = self.relu(temp)
        x = x + temp
        x = self.cov(x, edges)
        return self.relu(x)


class GlobalGraphNet(nn.Module):
    def __init__(self, cat_len=400, poi_len=38333, cat_dim=100, poi_dim=300, gcn_channel=64, gcn_layers=5, graph_out_dim=128,
                 gps_dim=400, lat_len=7128, long_len=6394):
        super(GlobalGraphNet, self).__init__()
        self.gcn_channel = gcn_channel
        self.gcn_layers = gcn_layers
        self.cat_emb = nn.Embedding(cat_len, cat_dim)
        self.poi_emb = nn.Embedding(poi_len, poi_dim)
        self.lat_emb = nn.Embedding(lat_len, gps_dim // 2)
        self.long_emb = nn.Embedding(long_len, gps_dim // 2)
        self.cov_in = GCNConv(cat_dim + poi_dim + gps_dim, gcn_channel)
        self.gcn_net = self.build_gcn_net()
        self.cov_out = GCNConv(gcn_channel, 1)
        self.relu = nn.LeakyReLU()
        self.fc_layer = nn.Sequential(
            nn.Linear(poi_len - 1, 128),
            nn.LeakyReLU(),
            nn.Linear(128, graph_out_dim),
        )

    def forward(self, inputs, weight):
        feature = inputs.x
        edges = inputs.edge_index
        poi_feature = feature[:, 0: 1].int()
        cat_feature = feature[:, 1: 2].int()
        lat_feature = feature[:, 3:4].int()
        long_feature = feature[:, 4:].int()
        poi_feature = self.poi_emb(poi_feature)
        cat_feature = self.cat_emb(cat_feature)
        lat_feature = self.lat_emb(lat_feature)
        long_feature = self.long_emb(long_feature)
        poi_feature = poi_feature.reshape(poi_feature.shape[0], -1)
        cat_feature = cat_feature.reshape(cat_feature.shape[0], -1)
        lat_feature = lat_feature.reshape(lat_feature.shape[0], -1)
        long_feature = long_feature.reshape(long_feature.shape[0], -1)
        feature = torch.cat((poi_feature, cat_feature, lat_feature, long_feature), dim=1)
        feature = self.relu(self.cov_in(feature, edges))
        for i in range(self.gcn_layers):
            feature = self.gcn_net[i](feature, edges)
        feature = self.relu(self.cov_out(feature, edges))
        feature = feature.reshape(-1)
        output = self.fc_layer(feature)
        return output

    def build_gcn_net(self):
        gcn_net = nn.ModuleList()
        for i in range(self.gcn_layers):
            gcn_net.append(GcnUnit(self.gcn_channel))
        return gcn_net


class GlobalDistNet(nn.Module):
    def __init__(self,  poi_dim=128, poi_len=38333, graph_features=544, gcn_channel=128, gcn_layers=4, graph_out_dim=128):
        super(GlobalDistNet, self).__init__()
        self.poi_len = poi_len
        self.gcn_layers = gcn_layers
        self.gcn_channel = gcn_channel
        self.emb = nn.Embedding(poi_len, poi_dim)
        self.gcn_net = self.build_gcn_net()
        self.cov_in = GCNConv(poi_dim * graph_features // 2 + graph_features // 2, gcn_channel)
        self.cov_out = GCNConv(gcn_channel, 1)
        self.relu = nn.LeakyReLU()
        self.fc_layer = nn.Sequential(
            nn.Linear(poi_len - 1, 128),
            nn.LeakyReLU(),
            nn.Linear(128, graph_out_dim),
        )

    def forward(self, inputs, mask, weight):
        feature = inputs.x
        edges = inputs.edge_index
        poi = torch.masked_select(feature, mask)
        distance = torch.masked_select(feature, ~mask)
        poi = poi.reshape(self.poi_len - 1, -1).int()
        distance = distance.reshape(self.poi_len - 1, -1)
        emb_poi = self.emb(poi)
        feature = torch.cat((emb_poi.reshape(emb_poi.shape[0], -1), distance), dim=1)
        feature = self.relu(self.cov_in(feature, edges))
        for i in range(self.gcn_layers):
            feature = self.gcn_net[i](feature, edges)
        feature = self.relu(self.cov_out(feature, edges))
        feature = feature.reshape(-1)
        output = self.fc_layer(feature)
        return output

    def build_gcn_net(self):
        gcn_net = nn.ModuleList()
        for i in range(self.gcn_layers):
            gcn_net.append(GcnUnit(self.gcn_channel))
        return gcn_net


class UserGraphNet(nn.Module):
    def __init__(self, cat_len=400, poi_len=5099, cat_dim=100, poi_dim=300, gcn_channel=128, gcn_layers=3, gps_dim=400,
                 node_len=714, graph_out_dim=128, lat_len=7128, long_len=6394):
        super(UserGraphNet, self).__init__()
        self.gcn_layers = gcn_layers
        self.gcn_channel = gcn_channel
        self.cat_emb = nn.Embedding(cat_len, cat_dim)
        self.poi_emb = nn.Embedding(poi_len, poi_dim)
        self.lat_emb = nn.Embedding(lat_len, gps_dim // 2)
        self.long_emb = nn.Embedding(long_len, gps_dim // 2)
        self.gcn_net = self.build_gcn_net()
        self.cov_in = GCNConv(cat_dim + poi_dim + gps_dim, gcn_channel)
        self.cov_out = GCNConv(gcn_channel, 1)
        self.relu = nn.LeakyReLU()
        self.fc_layer = nn.Sequential(
            nn.Linear(node_len, 128),
            nn.LeakyReLU(),
            nn.Linear(128, graph_out_dim),
        )

    def forward(self, feature, edges, weight):
        raw_batch = feature.shape[0]
        poi_feature = feature[:, :, 0: 1].int()
        cat_feature = feature[:, :, 1: 2].int()
        lat_feature = feature[:, :, 3:4].int()
        long_feature = feature[:, :, 4:].int()
        poi_feature = self.poi_emb(poi_feature)
        cat_feature = self.cat_emb(cat_feature)
        lat_feature = self.lat_emb(lat_feature)
        long_feature = self.long_emb(long_feature)
        poi_feature = poi_feature.reshape(poi_feature.shape[0], poi_feature.shape[1], -1)
        cat_feature = cat_feature.reshape(cat_feature.shape[0], cat_feature.shape[1], -1)
        lat_feature = lat_feature.reshape(lat_feature.shape[0], lat_feature.shape[1], -1)
        long_feature = long_feature.reshape(long_feature.shape[0], long_feature.shape[1], -1)
        feature = torch.cat((poi_feature, cat_feature, lat_feature, long_feature), dim=2)
        user_graph_data = self.build_graph_data(feature, edges, weight)
        x, edges, weight, batch = user_graph_data.x, user_graph_data.edge_index, user_graph_data.edge_attr, user_graph_data.batch
        weight = weight[:, 1:2]
        feature = self.relu(self.cov_in(x, edges))
        for i in range(self.gcn_layers):
            feature = self.gcn_net[i](feature, edges)
        feature = self.relu(self.cov_out(feature, edges))
        feature = feature.reshape(raw_batch, -1)
        output = self.fc_layer(feature)
        return output

    def build_graph_data(self, feature, edges, weight):
        data_list = [Data(t, j, z) for t, j, z in zip(feature, edges, weight)]
        graph_loader = DataLoader(data_list, feature.shape[0])
        for _, user_graph_data in enumerate(graph_loader):
            return user_graph_data

    def build_gcn_net(self):
        gcn_net = nn.ModuleList()
        for i in range(self.gcn_layers):
            gcn_net.append(AGCNUnit(self.gcn_channel))
        return gcn_net


class UserHistoryNet(nn.Module):
    def __init__(self, cat_len=400, poi_len=5099, user_len=1083, embed_size_user=20, embed_size_poi=300,
                 embed_size_cat=100, embed_size_hour=20, hidden_size=128, lstm_layers=3, history_out_dim=128):
        super(UserHistoryNet, self).__init__()
        self.embed_user = nn.Embedding(user_len, embed_size_user)
        self.embed_poi = nn.Embedding(poi_len, embed_size_poi)
        self.embed_cat = nn.Embedding(cat_len, embed_size_cat)
        self.embed_past = nn.Embedding(cat_len + poi_len + user_len, embed_size_user + embed_size_cat + embed_size_poi)
        self.embed_hour = nn.Embedding(24, embed_size_hour)
        self.embed_week = nn.Embedding(7, 7)
        self.gru_poi = nn.GRU(embed_size_user + embed_size_poi + embed_size_hour + 7, hidden_size, lstm_layers, dropout=0.5, batch_first=True)
        self.gru_cat = nn.GRU(embed_size_user + embed_size_cat + embed_size_hour + 7, hidden_size, lstm_layers, dropout=0.5, batch_first=True)
        self.gru_past = nn.GRU(embed_size_user + embed_size_poi + embed_size_cat + embed_size_hour + 7, hidden_size, lstm_layers, dropout=0.5, batch_first=True)
        self.poi_fc = nn.Linear(hidden_size, history_out_dim)
        self.cat_fc = nn.Linear(hidden_size, history_out_dim)
        self.out_w_poi = Parameter(torch.Tensor([0.5]).repeat(poi_len), requires_grad=True)
        self.out_w_cat = Parameter(torch.Tensor([0.5]).repeat(poi_len), requires_grad=True)
        self.out_w_hist = Parameter(torch.Tensor([0.33]).repeat(poi_len), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, neighbor):
        poi_feature = torch.cat((inputs[:, :, 0: 2], inputs[:, :, 3:]), dim=2)
        cat_feature = torch.cat((inputs[:, :, 0:1], inputs[:, :, 2:]), dim=2)
        poi_out = self.get_output(poi_feature, self.embed_poi, self.embed_user, self.embed_hour, self.embed_week, self.gru_poi, self.poi_fc)
        cat_out = self.get_output(cat_feature, self.embed_cat, self.embed_user, self.embed_hour, self.embed_week, self.gru_cat, self.cat_fc)
        out_w_poi = self.out_w_poi[inputs[:, :, 0: 1].long()]
        out_w_cat = self.out_w_cat[inputs[:, :, 0: 1].long()]
        poi_out = torch.mul(poi_out, out_w_poi)
        cat_out = torch.mul(cat_out, out_w_cat)
        embed_neighbor = self.embed_poi(neighbor.int())
        return poi_out + cat_out, embed_neighbor.reshape(embed_neighbor.shape[0], 1, -1)

    def get_output(self, inputs, embed_id, embed_user, embed_hour, embed_week, lstm, fc):
        b = inputs.shape[0]
        user_feature = inputs[:, :, 0: 1].int()
        id_feature = inputs[:, :, 1: 2].int()
        hour_feature = inputs[:, :, -2: -1].int()
        week_feature = inputs[:, :, -1:].int()
        emb_user = embed_user(user_feature.reshape(b, -1))
        emb_id = embed_id(id_feature.reshape(b, -1))
        emb_hour = embed_hour(hour_feature.reshape(b, -1))
        emb_week = embed_week(week_feature.reshape(b, -1))
        features = torch.cat((emb_user, emb_id, emb_hour, emb_week), dim=2)
        output, _ = lstm(features)
        return fc(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, poi_len):
        super(Attention, self).__init__()
        poi_len = 1024
        self.linear1 = nn.Linear(poi_len, poi_len)
        self.linear2 = nn.Linear(poi_len, poi_len)
        self.linear3 = nn.Linear(poi_len, poi_len)
        self.softmax = nn.LogSoftmax(-1)

    def forward(self, user_feature, global_graph_feature, global_dist_feature):
        q = self.linear1(user_feature)
        k = self.linear2(user_feature)
        v = self.linear3(user_feature)
        d_k = k.size(-2)
        qk = torch.bmm(q.transpose(2, 1), k)
        qk = qk / d_k
        graph_feature = torch.bmm(global_graph_feature.transpose(2, 1), global_dist_feature)
        att_value = torch.mul(graph_feature, qk)
        att_value = self.future_mask(att_value)
        att_value = self.softmax(att_value)
        output = torch.bmm(v, att_value)
        return output

    def future_mask(self, inputs):
        padding_num = float('-inf')
        diag_vals = torch.ones_like(inputs[0, :, :])  # (L_in, C_in)
        tril = torch.tril(diag_vals)  # (L_in, C_in)
        future_masks = tril.unsqueeze(0).repeat(inputs.size(0), 1, 1)

        paddings = (torch.ones_like(future_masks) * padding_num)
        outputs = torch.where(torch.eq(future_masks, 0.0), paddings, inputs)
        return outputs


class TransformerModel(nn.Module):
    def __init__(self, embed_dim, dropout, tran_head, tran_hid, tran_layers, poi_len, attention_layers=3):
        super(TransformerModel, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embed_dim, tran_head, tran_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, tran_layers)
        self.embed_size = embed_dim
        self.decoder_poi = nn.Linear(embed_dim, poi_len)
        self.init_weights()
        self.w_g_graph = Parameter(torch.Tensor([0.25]), requires_grad=True)
        self.w_g_dist = Parameter(torch.Tensor([0.25]), requires_grad=True)
        self.w_u_graph = Parameter(torch.Tensor([0.25]), requires_grad=True)
        self.w_u_nei = Parameter(torch.Tensor([0.25]), requires_grad=True)
        self.attention = Attention(poi_len)
        in_dim = 2000
        self.linear = nn.Linear(in_dim, 1024)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, user_history_feature, global_graph_feature, global_dist_feature, user_graph_feature, src_mask, embed_neighbor):
        embed_neighbor = self.linear(embed_neighbor)
        inputs = (torch.mul(global_graph_feature, self.w_g_graph) + torch.mul(global_dist_feature,  self.w_g_dist) +
                  torch.mul(user_graph_feature,  self.w_u_graph) + torch.mul(embed_neighbor, self.w_u_nei))
        inputs = inputs * math.sqrt(self.embed_size)
        inputs = self.pos_encoder(inputs)
        x = self.transformer_encoder(inputs)
        graph_feature = self.decoder_poi(x)
        out_put = graph_feature + user_history_feature
        return out_put

    def test(self, user_history_feature, global_graph_feature, global_dist_feature, user_graph_feature):
        user_feature = user_history_feature + user_graph_feature
        user_feature = self.linear(user_feature)
        output = self.attention(user_feature, global_graph_feature, global_dist_feature)
        output = self.fc_temp(output)
        return output


class GlobalUserNet(nn.Module):
    def __init__(self):
        super(GlobalUserNet, self).__init__()
        global_graph_net = GlobalGraphNet()
        global_dist_net = GlobalDistNet()
        user_graph_net = UserGraphNet()
        user_history_net = UserHistoryNet()
        transformer = TransformerModel()


class Rnn(nn.Module):
    def __init__(self, cat_len=400, poi_len=5099, user_len=1083, embed_size_user=20, embed_size_poi=300,
                 embed_size_cat=100, embed_size_hour=20, hidden_size=128, lstm_layers=3, history_out_dim=128):
        super(Rnn, self).__init__()
        self.embed_user = nn.Embedding(user_len, embed_size_user)
        self.embed_poi = nn.Embedding(poi_len, embed_size_poi)
        self.embed_cat = nn.Embedding(cat_len, embed_size_cat)
        self.embed_past = nn.Embedding(cat_len + poi_len + user_len, embed_size_user + embed_size_cat + embed_size_poi)
        self.embed_hour = nn.Embedding(24, embed_size_hour)
        self.embed_week = nn.Embedding(7, 7)
        self.rnn = nn.RNN(embed_size_user + embed_size_poi + embed_size_hour + 7 + embed_size_cat, hidden_size)
        self.gru = nn.GRU(embed_size_user + embed_size_poi + embed_size_hour + 7 + embed_size_cat, hidden_size, lstm_layers, dropout=0.5, batch_first=True)
        self.lstm = nn.GRU(embed_size_user + embed_size_poi + embed_size_hour + 7 + embed_size_cat, hidden_size,
                          lstm_layers, dropout=0.5, batch_first=True)
        self.poi_fc = nn.Linear(hidden_size, history_out_dim)
        self.out_w_poi = Parameter(torch.Tensor([0.5]).repeat(poi_len), requires_grad=True)
        self.out_w_cat = Parameter(torch.Tensor([0.5]).repeat(poi_len), requires_grad=True)
        self.out_w_hist = Parameter(torch.Tensor([0.33]).repeat(poi_len), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        poi_feature = torch.cat((inputs[:, :, 0: 2], inputs[:, :, 3:]), dim=2)
        cat_feature = inputs[:, :, 2:3]
        poi_out = self.get_output(poi_feature, cat_feature, self.embed_poi, self.embed_cat, self.embed_user, self.embed_hour, self.embed_week,
                                  self.gru, self.poi_fc)
        return poi_out

    def get_output(self, inputs, cat, embed_id, embed_cat, embed_user, embed_hour, embed_week, lstm, fc):
        b = inputs.shape[0]
        user_feature = inputs[:, :, 0: 1].int()
        id_feature = inputs[:, :, 1: 2].int()
        hour_feature = inputs[:, :, -2: -1].int()
        week_feature = inputs[:, :, -1:].int()
        cat = cat.int()
        emb_cat = embed_cat(cat.reshape(b, -1))
        emb_user = embed_user(user_feature.reshape(b, -1))
        emb_id = embed_id(id_feature.reshape(b, -1))
        emb_hour = embed_hour(hour_feature.reshape(b, -1))
        emb_week = embed_week(week_feature.reshape(b, -1))
        features = torch.cat((emb_user, emb_cat, emb_id, emb_hour, emb_week), dim=2)
        output, _ = lstm(features)
        return fc(output)
