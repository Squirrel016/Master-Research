import torch
from torch import nn

NUM_STATIONS = 4
EMBEDDING_DIM = 4
SEQ_FEATURE_DIM = 3  # population, hour_of_day, day_of_week
CURRENT_HOUR_DIM = 2  # hour_of_day, day_of_week of the prediction hour


class CrowdLSTM(nn.Module):
    """LSTM with station embedding and current-hour context."""

    def __init__(
        self,
        num_stations: int = NUM_STATIONS,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_size: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.station_embedding = nn.Embedding(num_stations, embedding_dim)
        lstm_input_size = SEQ_FEATURE_DIM + embedding_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size + CURRENT_HOUR_DIM, 1)

    def forward(
        self,
        x_seq: torch.Tensor,
        station_id: torch.Tensor,
        current_hour: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_seq: (batch, seq_len, 3) — population, hour_of_day, day_of_week.
            station_id: (batch,) — station indices.
            current_hour: (batch, 2) — hour_of_day, day_of_week for the target hour.

        Returns:
            (batch,) — predicted normalized population.
        """
        station_emb = self.station_embedding(station_id)
        station_emb = station_emb.unsqueeze(1).expand(-1, x_seq.size(1), -1)
        lstm_input = torch.cat([x_seq, station_emb], dim=-1)

        lstm_out, _ = self.lstm(lstm_input)
        context = torch.cat([lstm_out[:, -1, :], current_hour], dim=-1)
        prediction = self.fc(context)
        return prediction.squeeze(-1)
