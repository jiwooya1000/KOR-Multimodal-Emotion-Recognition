import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_model import *
from audio_model import *


# Combined Multi-Head Attention Block

class MultiLayer(nn.Module):
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

    def forward(self, x, mask=None):
        output = self.MHA(x, x, x, mask)
        output = x + self.dropout1(output)
        output = self.layerNorm1(output)

        output_ = self.ffn(output)
        output = output + self.dropout2(output_)
        output = self.layerNorm2(output)

        return output


class MultiBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_head=6, inner_dim=3072, n_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim

        self.MB = nn.ModuleList([MultiLayer(hidden_dim=self.hidden_dim,
                                            num_head=self.num_head,
                                            inner_dim=self.inner_dim) for i in range(n_layers)])

    def forward(self, x, total_mask):
        for layer in self.MB:
            x = layer(x, total_mask)

        return x


# Combined Audio Model & Text Model

class EmoClassifier(nn.Module):
    def __init__(self, hidden_dim=768, num_head=6, inner_dim=3072, n_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.inner_dim = inner_dim
        self.n_layers = n_layers

        # Load Audio Model
        self.AudioModel = AudioRegressor(hidden_dim=768, inner_dim=1024, n_layers=7)  # 설정값 넣기
        self.AudioModel.load_state_dict(torch.load('audio_arousal.pt'))

        # Load Text Model
        self.TextModel = TextRegressor(n_layers=5)
        self.TextModel.load_state_dict(torch.load('text_valence.pt'))

        # 사전학습된 Audio Model & Text model Freeze
        for param in self.AudioModel.parameters():
            param.requires_grad_(False)

        for param in self.TextModel.parameters():
            param.requires_grad_(False)

        # 연결 후 Self Attention
        self.MBA = MultiBlock(self.hidden_dim, self.num_head, self.inner_dim, self.n_layers)

        # Final Classifier
        self.LSTM = nn.LSTM(input_size=self.hidden_dim, hidden_size=128, num_layers=2, bias=False, batch_first=True,
                          dropout=0.1)
        self.fc = nn.Linear(128, 5)

        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, audio, input_ids, token_type_ids, attention_mask):
        audio_mask = padding_mask_audio(audio[:, 0, :, :].squeeze(dim=1))
        text_mask = attention_mask
        total_mask = torch.gt(torch.cat([audio_mask, text_mask], dim=2), 0)

        # 사전 학습된 Audio Model -> Arousal과 Attention 처리된 Audio Feature 반환
        with torch.no_grad():
            arousal, audio_output = self.AudioModel(audio=audio)

        # 사전 학습된 Text Model -> Valence와 Attention 처리된 Text Feature 반환
        with torch.no_grad():
            valence, text_output = self.TextModel(input_ids=input_ids,
                                                  token_type_ids=token_type_ids,
                                                  attention_mask=attention_mask)

        # 멀티모달 Self Attention
        multi_feature = torch.cat([audio_output, text_output], dim=1)
        output = self.MBA(multi_feature, total_mask=total_mask)

        # 감정 예측
        h0 = torch.zeros(2, audio.size(0), 128).requires_grad_().to(device)
        c0 = torch.zeros(2, audio.size(0), 128).requires_grad_().to(device)

        out, _ = self.LSTM(output, (h0, c0))
        h_t = out[:, -1, :]
        emotion = self.fc(h_t)

        return emotion, arousal, valence
