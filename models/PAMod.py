import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin

        self.phase_embedding = nn.Embedding(self.cycle_len, self.seq_len)
        self.ampli_embedding = nn.Embedding(self.cycle_len, self.enc_in * self.seq_len)
        nn.init.xavier_normal_(self.phase_embedding.weight)
        nn.init.xavier_normal_(self.ampli_embedding.weight)

        self.model = Modulator(self.seq_len, self.seq_len, dropout=0.5)

        self.output_proj = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )

    def forward(self, x, cycle_index):
        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True).detach()
            seq_var = torch.var(x, dim=1, keepdim=True).detach() + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # b,s,c -> b,c,s
        x_input = x.permute(0, 2, 1)
        B, N, _ = x_input.shape

        phase_emb = self.phase_embedding(cycle_index.view(-1, 1).expand(B, N))
        ampli_emb = self.ampli_embedding(cycle_index).reshape(B, self.enc_in, self.seq_len)

        hidden = self.model(x_input, ampli_emb, phase_emb)

        output = self.output_proj(hidden + x_input)
        output = output.permute(0, 2, 1)

        # instance denorm
        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output
    

class Modulator(nn.Module):
    def __init__(self, dim_in, dim_out, expand_ratio=4, dropout=0.5):
        super(Modulator, self).__init__()
        hidden_dim = int(dim_in * expand_ratio)
        
        self.W1 = nn.Linear(dim_in, hidden_dim)
        self.W2 = nn.Linear(dim_in, hidden_dim)
        self.W3 = nn.Linear(dim_in, hidden_dim)
        self.W4 = nn.Linear(hidden_dim, dim_out)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, query1, query2):
        return self.W4(self.dropout(F.relu(self.W1(x)) * self.W2(query1)) + self.W3(query2))