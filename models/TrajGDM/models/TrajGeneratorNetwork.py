import math

import torch
import torch.nn as nn
from torch.backends.cuda import SDPBackend, sdp_kernel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.checkpoint import (
    checkpoint,  # For a more effecient VRAM but at the expense of more computation time
)

backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class TrajGeneratorNetwork(torch.nn.Module):
    def __init__(
        self,
        num_location,
        location_embedding,
        lstm_hidden,
        maxi,
        device=torch.device("cuda:1"),
        input_len=12,
        num_head=2,
        TrajEncoder_layers=1,
        TrajGenerator_LSTMlayers=1,
        TrajGenerator_Translayers=1,
        TrajDecoder_layers=1,
    ):
        """
        Intialize the model
        :param num_location: number of unique locations (grid cells)
        :param location_embedding: embedding dimension for each locations
        :param lstm_hidden: hiden size for LSTM layers
        :param maxi: grid size (e.g: 110, 27)
        :param input_len: Length of input trajectory
        :num_head: Number of attention heads in transformers
        """
        super(TrajGeneratorNetwork, self).__init__()
        self.num_location = num_location
        self.model_device = device
        self.input_len = input_len
        self.loc_size = location_embedding
        if maxi == 110 or maxi == 27:
            self.expand_num_loc = (
                num_location + maxi * 4
            )  # accounting for adjacent locations
        else:
            self.expand_num_loc = num_location  # not accounting for adjacent locations

        self.num_head = num_head

        # added variable
        self.pad_tocken = 0

        # Encode diffusion timesteps
        self.temporal_embed = TimeEmbedding(location_embedding)  # .double()
        self.time_embed = TimeEmbedding(location_embedding)  # .double()

        # four positional encodings for different modules (like transformerd)
        self.pos_encoding1 = PositionalEncoding(
            num_features=location_embedding, dropout=0.1, max_len=self.input_len * 2
        )
        self.pos_encoding2 = PositionalEncoding(
            num_features=lstm_hidden, dropout=0.1, max_len=self.input_len * 2
        )
        self.pos_encoding3 = PositionalEncoding(
            num_features=lstm_hidden, dropout=0.1, max_len=self.input_len * 2
        )
        self.pos_encoding4 = PositionalEncoding(
            num_features=lstm_hidden, dropout=0.1, max_len=self.input_len * 2
        )
        self.start_embed = torch.nn.Embedding(
            num_embeddings=2, embedding_dim=location_embedding
        )  # start token

        self.location_embed = torch.nn.Embedding(
            num_embeddings=self.expand_num_loc,
            embedding_dim=self.loc_size,
            padding_idx=self.pad_tocken,
        )  # for each location and its neighbors

        if maxi == 110 or maxi == 27:
            self.spatial_embed = nn.Linear(
                location_embedding * 5, location_embedding
            )  # combine center and adjacent embeddings

        self.TrajEncoder_lstm = torch.nn.LSTM(
            input_size=self.loc_size,
            hidden_size=location_embedding,
            num_layers=TrajEncoder_layers,
            bidirectional=False,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(location_embedding)
        self.layer_norm2 = nn.LayerNorm(lstm_hidden)
        self.layer_norm3 = nn.LayerNorm(lstm_hidden)
        self.layer_norm4 = nn.LayerNorm(lstm_hidden)

        self.TrajGenerator_encoderLSTM = torch.nn.LSTM(
            input_size=location_embedding,
            hidden_size=lstm_hidden,
            num_layers=TrajGenerator_LSTMlayers,
            bidirectional=False,
            batch_first=True,
        )  # 2

        self.TrajGenerator_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=location_embedding,
                nhead=self.num_head,
                dim_feedforward=lstm_hidden,
                dropout=0.1,
                activation="relu",
                batch_first=True,
            ),
            num_layers=TrajGenerator_Translayers,
        )

        self.TrajGenerator_decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=location_embedding,
                nhead=self.num_head,
                dim_feedforward=lstm_hidden,
                dropout=0.1,
                activation="relu",
                batch_first=True,
            ),
            num_layers=TrajGenerator_Translayers,
        )

        self.TrajGenerator_decoderLSTM = torch.nn.LSTM(
            input_size=location_embedding,
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )  # 2

        self.TrajDecoder_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=location_embedding,
                nhead=self.num_head,
                dim_feedforward=lstm_hidden,
                dropout=0.1,
                activation="relu",
                batch_first=True,
            ),
            num_layers=TrajDecoder_layers,
        )

        self.TrajDecoder_decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=location_embedding,
                nhead=self.num_head,
                dim_feedforward=lstm_hidden,
                dropout=0.1,
                activation="relu",
                batch_first=True,
            ),
            num_layers=TrajDecoder_layers,
        )

        self.fc_layer = torch.nn.Linear(
            in_features=lstm_hidden, out_features=lstm_hidden
        )
        self.determintric_function = nn.Linear(lstm_hidden, self.num_location)

        # self = self.float() #make all layers in float 32

    def _lstm(self, lstm_module, x, lengths=None, enable_checkpoint=True):
        """Helper to LSTM operations (padded data)."""

        def lstm_forward(input_data, lens):
            if lens is not None:
                packed = pack_padded_sequence(
                    input_data, lens.cpu(), batch_first=True, enforce_sorted=False
                )
                packed_out, _ = lstm_module(packed)
                unpacked, _ = pad_packed_sequence(
                    packed_out, batch_first=True, total_length=input_data.shape[1]
                )
                return unpacked
            out, _ = lstm_module(input_data)
            return out

        if enable_checkpoint:
            return checkpoint(lstm_forward, x, lengths, use_reentrant=False)
        else:
            return lstm_forward(x, lengths)

    def TrajGenerator(self, xt, time, mask=None, lengths=None):
        """
        Forward process
        xt: the sequence batch  (batch, seq_len, embedding_dim) or (batch, seq_len) if not embedded yet
        time: (batch,)
        lengths: (batch,) actual lengths before padding
        """

        xt = xt.float()
        max_len = xt.shape[1]

        t = self.temporal_embed(time.float())

        src_padding_mask = ~(mask) if mask is not None else None
        xt_t = torch.unsqueeze(t, dim=1).repeat(1, max_len, 1)
        et = self.layer_norm(xt + xt_t)
        if lengths is not None:
            residual = et

            et_unpacked = self._lstm(
                self.TrajGenerator_encoderLSTM, et, lengths, enable_checkpoint=False
            )

            et = self.layer_norm2(
                et_unpacked + residual
            )  # Residual connection (only on valid positions)
            et = self.pos_encoding1(et)

            with sdp_kernel(backends):
                if False:
                    et = checkpoint(
                        self.TrajGenerator_encoder,
                        et,
                        src_key_padding_mask=src_padding_mask,
                        use_reentrant=False,  # Recommended for determinism and efficiency
                    )
                else:
                    et = self.TrajGenerator_encoder(
                        et, src_key_padding_mask=src_padding_mask
                    )
            et = et.contiguous()
            decoder_input = self.start_embed(
                torch.zeros(size=(et.shape[0], 1), device=et.device, dtype=torch.long)
            )
            output = torch.zeros_like(et, dtype=torch.float)  # double

        else:
            residual = et
            et, (hn, cn) = self.TrajGenerator_encoderLSTM(et)
            et = self.layer_norm2(et + residual)
            et = self.pos_encoding1(et)
            with sdp_kernel(backends):
                if False:
                    et = checkpoint(self.TrajGenerator_encoder, et, use_reentrant=False)
                else:
                    et = self.TrajGenerator_encoder(et)

        decoder_input = self.start_embed(
            torch.zeros(size=(et.shape[0], 1), device=et.device, dtype=torch.long)
        )

        # current_input_emb = self.start_embed(
        #     torch.zeros(size=(et.shape[0], 1), device=et.device, dtype=torch.long)
        # )

        # 1. Pre-calculate all positional encodings for the whole sequence length
        # This avoids calling self.pos_encoding2(decoder_input) N times
        # full_pos_enc = self.pos_encoding2.P[:, : et.shape[1] + 1, :].to(et.device)

        output = torch.zeros_like(xt, dtype=torch.float)
        with sdp_kernel(backends):
            for i in range(et.shape[1]):
                if i == 0:
                    decoder_input = self.pos_encoding2(decoder_input)
                else:
                    previous = decoder_input[:, :-1, :]
                    decoder_input = self.pos_encoding2(decoder_input)
                    decoder_input[:, :-1, :] = previous

                # step_input = (
                #     current_input_emb + full_pos_enc[:, : current_input_emb.size(1), :]
                # )
                # self.eval()  # Disables dropout
                # with torch.no_grad():
                #     print("are they the same?", step_input == decoder_input)
                #     if step_input != decoder_input:
                # raise ValueError
                tgt_mask = self.generate_square_subsequent_mask(
                    decoder_input.size(1)
                ).to(decoder_input.device)

                if self.training:
                    decoder_output = checkpoint(
                        self.TrajGenerator_decoder,
                        decoder_input,
                        et,
                        tgt_mask,
                        use_reentrant=False,
                    )
                else:
                    decoder_output = self.TrajGenerator_decoder(
                        decoder_input, et, tgt_mask
                    )  # ,tgt_mask
                output[:, i, :] = decoder_output[:, i, :]
                decoder_input = torch.cat(
                    (decoder_input, decoder_output[:, i, :].unsqueeze(1)), dim=1
                )
            output = self.fc_layer(output)

        if mask is not None:
            if not self.training:
                if output.shape[1] < mask.shape[1]:
                    print("we're clipping")
                    pad_len = mask.shape[1] - output.shape[1]
                    pad_tensor = torch.zeros(
                        output.shape[0], pad_len, output.shape[2], device=output.device
                    )
                    output = torch.cat([output, pad_tensor], dim=1)

            output = output.masked_fill(src_padding_mask.unsqueeze(-1), 0.0)
        return output

    def TrajDecoder(self, hidden_repr, src_padding_mask=None):
        src_mask = self.generate_square_subsequent_mask(hidden_repr.size(1)).to(
            hidden_repr.device
        )
        hidden_repr = self.pos_encoding3(hidden_repr)  # 1 3
        with sdp_kernel(backends):
            if False:
                hidden_repr = checkpoint(
                    self.TrajDecoder_encoder,
                    hidden_repr,
                    mask=src_mask,
                    src_key_padding_mask=src_padding_mask,
                    use_reentrant=False,
                )
            else:
                hidden_repr = self.TrajDecoder_encoder(
                    hidden_repr, mask=src_mask, src_key_padding_mask=src_padding_mask
                )

        decoder_input = self.location_embed(
            torch.ones(
                size=(hidden_repr.shape[0], hidden_repr.shape[1]),
                device=hidden_repr.device,
                dtype=torch.long,
            )
        )

        decoder_input = self.pos_encoding4(decoder_input)  # 2 4
        tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(1)).to(
            decoder_input.device
        )
        with sdp_kernel(backends):
            if False:
                decoder_output = checkpoint(
                    self.TrajDecoder_decoder,
                    decoder_input,
                    hidden_repr,
                    tgt_mask,
                    tgt_key_padding_mask=src_padding_mask,
                    memory_key_padding_mask=src_padding_mask,
                    use_reentrant=False,
                )
            else:
                decoder_output = self.TrajDecoder_decoder(
                    decoder_input,
                    hidden_repr,
                    tgt_mask,
                    tgt_key_padding_mask=src_padding_mask,  # If needed
                    memory_key_padding_mask=src_padding_mask,
                )

        outputs = self.determintric_function(decoder_output)
        # nomiates = torch.argmax(outputs,dim=-1)

        if src_padding_mask is not None:
            outputs = outputs.masked_fill(src_padding_mask.unsqueeze(-1), 0.0)
        return outputs  # ,nomiates

    def TrajEncoder(self, sequence, lengths=None):
        if lengths is not None:
            # Ignore padding in lstm
            packed_seq = pack_padded_sequence(
                sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, (hn, cn) = self.TrajEncoder_lstm(packed_seq)
            output, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=sequence.shape[1]
            )
            return output
        else:
            sequence, (hn, cn) = self.TrajEncoder_lstm(sequence)
            return sequence

    def LocationEncoder(self, locs, lab, maxi, pad_token=0):
        if pad_token is not None:
            mask = locs != pad_token
            locs_safe = locs.clone().long()
            locs_safe[~mask] = 0  # reset padding to zero

        if maxi == 110 or maxi == 27:
            # print("Embedding size:", self.location_embed.num_embeddings)
            # print("Locs max:", locs.max().item(), "min:", locs.min().item())
            # print("Locs+maxi max:", (locs + maxi).max().item())
            locs = locs_safe.long()  # added this (wich double to float32)
            west = self.location_embed((locs_safe + maxi))
            east = self.location_embed((locs_safe + 2 + maxi))
            north = self.location_embed(locs_safe + 1)
            south = self.location_embed(locs_safe + 1 + (2 * maxi))
            center = self.location_embed(locs_safe + 1 + maxi)
            sequence = self.spatial_embed(
                torch.cat((center * lab, south, north, east, west), dim=-1)
            )  # Each location is represented by concatenating its embeddings, the embedding of its neighbors
            sequence[~mask] = 0
            return sequence
        else:
            sequence = self.location_embed(locs_safe)
            return sequence

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    @property
    def get_device(self):
        return self.model_device


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(torch.nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.dim = 4
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb


class PositionalEncoding(nn.Module):
    def __init__(self, num_features, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_features))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_features, 2, dtype=torch.float32) / num_features
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)
