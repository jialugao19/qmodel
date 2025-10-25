import torch
import torch.nn as nn

class GRU1(nn.Module):
    def __init__(self, config):
        super(GRU1, self).__init__()
        self.config = config

        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.gru_hidden_dim,
            num_layers=config.gru_num_layers,
            bidirectional=config.gru_bidirectional,
            dropout=config.gru_dropout if config.gru_num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(config.gru_hidden_dim, config.output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[:, -1, :])
        return output

class GRU2(nn.Module):
    def __init__(self, config):
        super(GRU2, self).__init__()
        self.config = config

        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.gru_hidden_dim,
            num_layers=config.gru_num_layers,
            bidirectional=config.gru_bidirectional,
            dropout=config.gru_dropout if config.gru_num_layers > 1 else 0,
            batch_first=True
        )

        gru_output_dim = config.gru_hidden_dim * config.window_size
        if config.gru_bidirectional:
            gru_output_dim *= 2

        self.decoder = nn.Sequential(
            nn.Linear(gru_output_dim, config.dense_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dense_dropout),
            nn.Linear(config.dense_hidden_dim, config.dense_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dense_dropout),
            nn.Linear(config.dense_hidden_dim, config.output_dim)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.decoder(gru_out.flatten(start_dim=1))
        return output

class GRU3(nn.Module):
    def __init__(self, config):
        super(GRU3, self).__init__()
        self.config = config

        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.gru_hidden_dim,
            num_layers=config.gru_num_layers,
            bidirectional=config.gru_bidirectional,
            dropout=config.gru_dropout if config.gru_num_layers > 1 else 0,
            batch_first=True
        )

        gru_output_dim = config.gru_hidden_dim
        if config.gru_bidirectional:
            gru_output_dim *= 2

        self.decoder = nn.Sequential(
            nn.Linear(gru_output_dim, config.dense_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dense_dropout),
            nn.Linear(config.dense_hidden_dim, config.dense_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dense_dropout),
            nn.Linear(config.dense_hidden_dim, config.output_dim)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.decoder(gru_out[:, -1, :])
        return output

class ConvGRU1(nn.Module):
    def __init__(self, config):
        super(ConvGRU1, self).__init__()
        self.config = config

        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=config.input_dim,
                out_channels=config.conv_out_channels,
                kernel_size=config.conv_kernel_size,
                stride=config.conv_stride,
                padding=config.conv_kernel_size // 2
            ),
            nn.ReLU(),
            nn.Dropout(config.conv_dropout),
            nn.Conv1d(
                in_channels=config.conv_out_channels,
                out_channels=config.conv_out_channels,
                kernel_size=config.conv_kernel_size,
                stride=config.conv_stride,
                padding=config.conv_kernel_size // 2
            ),
            nn.ReLU(),
            nn.Dropout(config.conv_dropout)
        )

        self.gru = nn.GRU(
            input_size=config.conv_out_channels,
            hidden_size=config.gru_hidden_dim,
            num_layers=config.gru_num_layers,
            bidirectional=config.gru_bidirectional,
            dropout=config.gru_dropout if config.gru_num_layers > 1 else 0,
            batch_first=True
        )

        gru_output_dim = config.gru_hidden_dim * (2 if config.gru_bidirectional else 1)
        self.fc = nn.Linear(gru_output_dim, config.output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        conv_out = self.conv1d(x)
        conv_out = conv_out.transpose(1, 2)
        gru_out, _ = self.gru(conv_out)
        output = self.fc(gru_out[:, -1, :])
        return output

