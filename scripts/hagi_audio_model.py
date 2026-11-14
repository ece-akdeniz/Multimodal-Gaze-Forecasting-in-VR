import torch
import torch.nn as nn


class HAGIAudioRegression(nn.Module):
    def __init__(
        self,
        seq_len=10,
        head_dim=3,
        audio_dim=4,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        audio_coord_dim=None,
        audio_validity_dim=0,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.head_dim = head_dim
        self.audio_dim = audio_dim
        self.audio_coord_dim = audio_coord_dim if audio_coord_dim is not None else audio_dim
        self.audio_validity_dim = audio_validity_dim

        self.gru = nn.GRU(
            input_size=head_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.audio_mlp = nn.Sequential(
            nn.Linear(audio_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def _split_audio(self, x):
        head_end = self.seq_len * self.head_dim
        head_features = x[:, :head_end]
        audio_start = head_end
        audio_end = audio_start + self.audio_coord_dim

        if self.audio_validity_dim > 0:
            audio_coords = x[:, audio_start:audio_end]
            audio_validity = x[:, audio_end : audio_end + self.audio_validity_dim]
            audio_features = torch.cat([audio_coords, audio_validity], dim=-1)
        else:
            audio_features = x[:, audio_start:]
            audio_validity = None

        return head_features, audio_features, audio_validity

    def forward(self, x):
        head_features, audio_features, _ = self._split_audio(x)
        head_features = head_features.view(x.shape[0], self.seq_len, self.head_dim)

        _, h = self.gru(head_features)
        head_embedding = h[-1]

        audio_embedding = self.audio_mlp(audio_features)
        fused = torch.cat([head_embedding, audio_embedding], dim=-1)
        y = self.head(fused)
        y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
        return y


class HAGIAudioGatedRegression(nn.Module):
    def __init__(
        self,
        seq_len=10,
        head_dim=3,
        audio_coord_dim=6,
        audio_validity_dim=3,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.head_dim = head_dim
        self.audio_coord_dim = audio_coord_dim
        self.audio_validity_dim = audio_validity_dim

        self.gru = nn.GRU(
            input_size=head_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.audio_mlp = nn.Sequential(
            nn.Linear(audio_coord_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.gate = nn.Sequential(
            nn.Linear(audio_validity_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        head_end = self.seq_len * self.head_dim
        head_features = x[:, :head_end]
        audio_coords = x[:, head_end : head_end + self.audio_coord_dim]
        audio_validity = x[:, head_end + self.audio_coord_dim : head_end + self.audio_coord_dim + self.audio_validity_dim]

        head_features = head_features.view(x.shape[0], self.seq_len, self.head_dim)

        _, h = self.gru(head_features)
        head_embedding = h[-1]

        audio_embedding = self.audio_mlp(audio_coords)
        gate = self.gate(audio_validity)
        gated_audio = audio_embedding * gate

        fused = torch.cat([head_embedding, gated_audio], dim=-1)
        y = self.head(fused)
        y = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
        return y


def build_audio_model(
    fusion,
    seq_len,
    audio_dim,
    audio_coord_dim=None,
    audio_validity_dim=0,
    hidden_dim=128,
    num_layers=2,
    dropout=0.2,
):
    if audio_coord_dim is None:
        audio_coord_dim = audio_dim

    if fusion == "gated":
        if audio_validity_dim <= 0:
            raise ValueError("gated fusion requires --audio-validity-dim > 0")
        return HAGIAudioGatedRegression(
            seq_len=seq_len,
            audio_coord_dim=audio_coord_dim,
            audio_validity_dim=audio_validity_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    mlp_audio_dim = audio_coord_dim + audio_validity_dim if audio_validity_dim > 0 else audio_dim
    return HAGIAudioRegression(
        seq_len=seq_len,
        audio_dim=mlp_audio_dim,
        audio_coord_dim=audio_coord_dim,
        audio_validity_dim=audio_validity_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
