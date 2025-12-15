"""
Vendorized minimal SGAN TrajectoryGenerator for inference.
Source: https://github.com/agrimgupta92/sgan (MIT License)
Adapted for modern PyTorch and device-agnostic execution.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def make_mlp(dim_list, activation="relu", batch_norm=True, dropout=0.0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type: str, device):
    if noise_type == "gaussian":
        return torch.randn(*shape, device=device)
    elif noise_type == "uniform":
        return torch.rand(*shape, device=device).sub_(0.5).mul_(2.0)
    raise ValueError(f'Unrecognized noise type "{noise_type}"')


class Encoder(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0):
        super().__init__()
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch, device):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim, device=device),
            torch.zeros(self.num_layers, batch, self.h_dim, device=device),
        )

    def forward(self, obs_traj, device):
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch, device)
        _, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h


class Decoder(nn.Module):
    def __init__(
        self,
        seq_len,
        embedding_dim=64,
        h_dim=128,
        mlp_dim=1024,
        num_layers=1,
        pool_every_timestep=True,
        dropout=0.0,
        bottleneck_dim=1024,
        activation="relu",
        batch_norm=True,
        pooling_type="pool_net",
        neighborhood_size=2.0,
        grid_size=8,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep and (pooling_type is not None)
        self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

        if self.pool_every_timestep:
            if pooling_type == "pool_net":
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                )
            elif pooling_type == "spool":
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size,
                    pool_dim=bottleneck_dim,
                )

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                decoder_h = state_tuple[0]
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
                decoder_h = torch.cat([decoder_h.view(-1, self.h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos
            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class PoolHiddenNet(nn.Module):
    def __init__(
        self,
        embedding_dim=64,
        h_dim=64,
        mlp_dim=1024,
        bottleneck_dim=1024,
        activation="relu",
        batch_norm=True,
        dropout=0.0,
    ):
        super().__init__()
        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims, activation=activation, batch_norm=batch_norm, dropout=dropout
        )
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim

    @staticmethod
    def repeat(tensor, num_reps):
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class SocialPooling(nn.Module):
    def __init__(
        self,
        h_dim=64,
        activation="relu",
        batch_norm=True,
        dropout=0.0,
        neighborhood_size=2.0,
        grid_size=8,
        pool_dim=None,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim or h_dim]
        self.mlp_pool = make_mlp(
            mlp_pool_dims, activation=activation, batch_norm=batch_norm, dropout=dropout
        )

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    @staticmethod
    def get_grid_locations(top_left, other_pos, grid_size, neighborhood_size):
        cell_x = torch.floor(((other_pos[:, 0] - top_left[:, 0]) / neighborhood_size) * grid_size)
        cell_y = torch.floor(((top_left[:, 1] - other_pos[:, 1]) / neighborhood_size) * grid_size)
        grid_pos = cell_x + cell_y * grid_size
        return grid_pos

    @staticmethod
    def repeat(tensor, num_reps):
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        pool_h = []
        grid_size = self.grid_size * self.grid_size
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))

            top_left, bottom_right = self.get_bounds(curr_end_pos)
            curr_end_pos_rep = curr_end_pos.repeat(num_ped, 1)
            top_left_rep = self.repeat(top_left, num_ped)
            bottom_right_rep = self.repeat(bottom_right, num_ped)

            grid_pos = self.get_grid_locations(
                top_left_rep, curr_end_pos_rep, self.grid_size, self.neighborhood_size
            ).type_as(seq_start_end)

            x_bound = ((curr_end_pos_rep[:, 0] >= bottom_right_rep[:, 0]) + (curr_end_pos_rep[:, 0] <= top_left_rep[:, 0]))
            y_bound = ((curr_end_pos_rep[:, 1] >= top_left_rep[:, 1]) + (curr_end_pos_rep[:, 1] <= bottom_right_rep[:, 1]))
            within_bound = x_bound + y_bound
            within_bound[0 :: num_ped + 1] = 1
            within_bound = within_bound.view(-1)

            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange(0, total_grid_size * num_ped, total_grid_size).type_as(seq_start_end)
            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos, curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h


class TrajectoryGenerator(nn.Module):
    def __init__(
        self,
        obs_len: int,
        pred_len: int,
        embedding_dim: int = 64,
        encoder_h_dim: int = 64,
        decoder_h_dim: int = 128,
        mlp_dim: int = 1024,
        num_layers: int = 1,
        noise_dim: Tuple[int, ...] = (0,),
        noise_type: str = "gaussian",
        noise_mix_type: str = "ped",
        pooling_type: Optional[str] = None,
        pool_every_timestep: bool = True,
        dropout: float = 0.0,
        bottleneck_dim: int = 1024,
        activation: str = "relu",
        batch_norm: bool = True,
        neighborhood_size: float = 2.0,
        grid_size: int = 8,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        if pooling_type and pooling_type.lower() == "none":
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = bottleneck_dim
        self.device = device

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size,
        )

        if pooling_type == "pool_net":
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )
        elif pooling_type == "spool":
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size,
            )

        if self.noise_dim and self.noise_dim[0] == 0:
            self.noise_dim = None
            self.noise_first_dim = 0
        else:
            self.noise_first_dim = self.noise_dim[0] if self.noise_dim else 0

        input_dim = encoder_h_dim + (bottleneck_dim if pooling_type else 0)
        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim]
            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == "global":
            noise_shape = (seq_start_end.size(0),) + self.noise_dim
        else:
            noise_shape = (_input.size(0),) + self.noise_dim

        z_decoder = user_noise if user_noise is not None else get_noise(noise_shape, self.noise_type, self.device)

        if self.noise_mix_type == "global":
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)
        return decoder_h

    def mlp_decoder_needed(self):
        return bool(self.noise_dim or self.pooling_type or self.encoder_h_dim != self.decoder_h_dim)

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None):
        batch = obs_traj_rel.size(1)
        final_encoder_h = self.encoder(obs_traj_rel, device=self.device)
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
            mlp_decoder_context_input = torch.cat([final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(-1, self.encoder_h_dim)

        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim, device=self.device)
        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]

        decoder_out = self.decoder(last_pos, last_pos_rel, state_tuple, seq_start_end)
        pred_traj_fake_rel, _ = decoder_out
        return pred_traj_fake_rel
