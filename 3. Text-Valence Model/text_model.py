import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from kobert import get_pytorch_kobert_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#### padding_mask_text, Self_Attention, MHA, FFN : audio_model.py와 동일

def padding_mask_text(text):
    return torch.all(text != 0, axis=2).unsqueeze(-2)


def Self_Attention(Q, K, V, mask=None):
    K_t = torch.transpose(K, -2, -1)
    KV = torch.matmul(Q, K_t)
    dim = Q.size()[-1]
    drop = nn.Dropout(p=0.1)

    score = KV / math.sqrt(dim)
    if mask is not None:
        mask = mask.unsqueeze(1)
        score = torch.masked_fill(score, ~mask, -1e9)

    score = drop(F.softmax(score, dim=-1))

    att_value = torch.matmul(score, V)

    return att_value, score


class MHA(nn.Module):
    def __init__(self, hidden_dim=128, num_head=8, dropout=0.1, device=device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.head_dim = hidden_dim // num_head
        self.scale = torch.sqrt(torch.FloatTensor()).to(device)

        self.Q_fc = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.K_fc = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.V_fc = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.Out_fc = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q_input, K_input, V_input, mask=None):
        batch_size = Q_input.size(0)

        Q = self.Q_fc(Q_input).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        K = self.K_fc(K_input).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)
        V = self.V_fc(V_input).view(batch_size, -1, self.num_head, self.head_dim).transpose(1, 2)

        att_value, score = Self_Attention(Q, K, V, mask=mask)
        att_value = att_value.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        output = self.Out_fc(att_value)

        return output


class FFN(nn.Module):
    def __init__(self, hidden_dim=768, inner_dim=3072, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim

        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        res = x
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        output = x + res

        return output


class TELayer1(nn.Module):
    def __init__(self, hidden_dim=768, num_head=6, inner_dim=3072):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim

        self.MHA = MHA(self.hidden_dim, self.num_head)
        self.layerNorm1 = nn.LayerNorm(self.hidden_dim)
        self.layerNorm2 = nn.LayerNorm(self.hidden_dim)
        self.ffn = FFN(self.hidden_dim, self.inner_dim)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, x, mask):
        output = self.MHA(x, x, x, mask)
        output = x + self.dropout1(output)
        output = self.layerNorm1(output)

        output_ = self.ffn(output)
        output = output + self.dropout2(output_)
        output = self.layerNorm2(output)

        return output


class TEBlock1(nn.Module):
    def __init__(self, hidden_dim=768, num_head=6, inner_dim=3072, n_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim

        self.TE1 = nn.ModuleList([TELayer1(hidden_dim=self.hidden_dim,
                                           num_head=self.num_head,
                                           inner_dim=self.inner_dim) for i in range(n_layers)])

    def forward(self, x):
        masking = padding_mask_text(x)

        for layer in self.TE1:
            x = layer(x, masking)

        return x


class TextRegressor(nn.Module):
    def __init__(self, hidden_dim=768, num_head=6, inner_dim=1024, n_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim
        self.n_layers = n_layers
        
        # 사전학습된 KoBERT 모델을 활용하여 Text Embedding 진행
        self.kobert, vocab = get_pytorch_kobert_model()
        for param in self.kobert.parameters():
            param.requires_grad = False

        # 임베딩 벡터에 대해 Self Attention 진행
        self.TEBlock1 = TEBlock1(self.hidden_dim, self.num_head, self.inner_dim, self.n_layers)
        
        # KoBERT를 통해 Positional Embedding이 되어있으므로 GRU를 통한 Valence 예측
        self.gru = nn.GRU(input_size=self.hidden_dim, hidden_size=64, num_layers=2, bias=False, batch_first=True,
                          dropout=0.1)
        self.fc = nn.Linear(64, 1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        self.kobert.eval()
        with torch.no_grad():
            text_output, _ = self.kobert(input_ids=input_ids.squeeze(dim=1),
                                         token_type_ids=token_type_ids.squeeze(dim=1),
                                         attention_mask=attention_mask.squeeze(dim=1))

        text_output = self.TEBlock1(text_output)

        h0 = torch.zeros(2, input_ids.size(0), 64).requires_grad_().to(device)

        out, h = self.gru(text_output, h0)
        h_t = out[:, -1, :]
        output = self.fc(h_t)

        return output, text_output
