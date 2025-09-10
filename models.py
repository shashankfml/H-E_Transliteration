import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import heapq


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.score(hidden, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        energy = F.relu(self.attn(torch.cat([h, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.unsqueeze(0).expand(encoder_outputs.size(0), -1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class Encoder(nn.Module):
    def __init__(self, num_encoder_tokens, hidden_dim, n_layers, dropout, encoder_embedding_dim=0, cell_type="LSTM", verbose=False):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.cell_type = cell_type
        self.embedding_dim = encoder_embedding_dim
        self.encoder_input_size = num_encoder_tokens
        self.verbose = verbose
        self.dropout = nn.Dropout(dropout)

        dropout = 0 if (n_layers == 1) else dropout

        if self.embedding_dim != 0:
            self.encoder_input_size = self.embedding_dim
            self.embedding = nn.Embedding(num_encoder_tokens, self.embedding_dim, padding_idx=num_encoder_tokens-1)

        if self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(self.encoder_input_size, hidden_dim, n_layers, dropout=dropout)
        elif self.cell_type == 'RNN':
            self.rnn = nn.RNN(self.encoder_input_size, hidden_dim, n_layers, dropout=dropout)
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(self.encoder_input_size, hidden_dim, n_layers, dropout=dropout)

    def forward(self, input):
        input = input.transpose(0, 1)

        if self.embedding_dim != 0:
            input = input.argmax(2)
            input = self.dropout(self.embedding(input))

        if self.verbose:
            print(f"Input shape after embedding: {input.shape}")

        outputs, hidden_cell = self.rnn(input)

        if self.verbose:
            print(f"Input shape: {input.shape}")
            print(f"Outputs shape: {outputs.shape}")
            print(f"Hidden/Cell state shape: {hidden_cell.shape}")

        return outputs, hidden_cell


class Decoder(nn.Module):
    def __init__(self, num_decoder_tokens, decoder_hidden_dim, n_layers, dropout, decoder_embedding_dim=0, 
                 cell_type='LSTM', atten=False, verbose=False):
        super(Decoder, self).__init__()
        
        self.output_dim = num_decoder_tokens
        self.decoder_hidden_dim = decoder_hidden_dim
        self.cell_type = cell_type
        self.attention = atten
        self.n_layers = n_layers
        self.decoder_embedding_dim = decoder_embedding_dim
        self.decoder_input = num_decoder_tokens
        self.verbose = verbose
        self.dropout = nn.Dropout(dropout)

        dropout = 0 if (n_layers == 1) else dropout

        if self.decoder_embedding_dim != 0:
            self.decoder_input = self.decoder_embedding_dim
            self.embedding = nn.Embedding(num_decoder_tokens, self.decoder_embedding_dim)

        if self.attention == False:
            if cell_type == 'LSTM':
                self.rnn = nn.LSTM(self.decoder_input, self.decoder_hidden_dim, n_layers, dropout=dropout)
            elif cell_type == 'RNN':
                self.rnn = nn.RNN(self.decoder_input, self.decoder_hidden_dim, n_layers, dropout=dropout)
            elif cell_type == 'GRU':
                self.rnn = nn.GRU(self.decoder_input, self.decoder_hidden_dim, n_layers, dropout=dropout)
            self.fc_out = nn.Linear(self.decoder_hidden_dim, self.output_dim)
        else:
            self.attention = Attention(self.decoder_hidden_dim)
            
            if cell_type == "LSTM":
                self.rnn = nn.LSTM(self.decoder_hidden_dim + self.decoder_input, decoder_hidden_dim, n_layers, dropout=dropout)
            elif cell_type == "RNN":
                self.rnn = nn.RNN(self.decoder_hidden_dim + self.decoder_input, decoder_hidden_dim, n_layers, dropout=dropout)
            elif cell_type == "GRU":
                self.rnn = nn.GRU(self.decoder_hidden_dim + self.decoder_input, decoder_hidden_dim, n_layers, dropout=dropout)
            self.fc_out = nn.Linear(self.decoder_hidden_dim * 2, self.output_dim)

    def forward(self, input, hidden_cell, encoder_states):
        if isinstance(hidden_cell, tuple):
            hidden = hidden_cell[0]
            cell = hidden_cell[1]
        else:
            hidden = hidden_cell

        if self.decoder_embedding_dim != 0:
            input = input.argmax(2)
            input = self.dropout(self.embedding(input))

        if self.attention == False:
            if self.cell_type == "LSTM":
                output, hidden = self.rnn(input, (hidden, cell))
            else:
                output, hidden = self.rnn(input, hidden)
            prediction = self.fc_out(output.squeeze(0))
            return prediction, hidden
        else:
            attn_weights = self.attention(hidden[-1], encoder_states)
            context = attn_weights.bmm(encoder_states.transpose(0, 1))
            context = context.transpose(0, 1)
            rnn_input = torch.cat([input, context], 2)

            if self.cell_type == "LSTM":
                output, hidden = self.rnn(rnn_input, (hidden, cell))
            else:
                output, hidden = self.rnn(rnn_input, hidden)
            
            output = output.squeeze(0)
            context = context.squeeze(0)
            output = self.fc_out(torch.cat([output, context], 1))

            if self.verbose:
                print(output.shape)
                print(hidden.shape)

            return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, max_source_length, max_target_length, target_char2int, num_decoder_tokens, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.atten = self.decoder.attention
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.target_chr2int = target_char2int
        self.num_decoder_tokens = num_decoder_tokens

        assert encoder.hidden_dim == decoder.decoder_hidden_dim, "Hidden dimensions of encoder and decoder must be equal"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers"

    def forward(self, src, trg, to_train, teacher_forcing_ratio=0.5, beam_width=3):
        if to_train:
            teacher_forcing_ratio = teacher_forcing_ratio
        else:
            teacher_forcing_ratio = 0

        trg = trg.transpose(0, 1)
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_output, hidden_cell = self.encoder(src)
        inp = trg[0, :]

        for t in range(1, trg_len):
            if self.atten == False:
                prediction, hidden_cell = self.decoder(inp.unsqueeze(0), hidden_cell, encoder_output)
            else:
                prediction, hidden_cell, attn_weights = self.decoder(inp.unsqueeze(0), hidden_cell, encoder_output)

            outputs[t] = prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            top1_one_hot = torch.zeros_like(prediction).to(self.device)
            top1_one_hot[:, top1] = 1.0
            inp = trg[t] if teacher_force else top1_one_hot

        return outputs

    def beam_search(self, src, beam_width=3, max_length=None):
        """Beam search decoding for inference"""
        if max_length is None:
            max_length = self.max_target_length
            
        batch_size = src.size(0)
        encoder_output, hidden_cell = self.encoder(src)
        
        # Start token (assuming index 0 is start token)
        start_token = torch.zeros(1, self.num_decoder_tokens).to(self.device)
        start_token[0, self.target_chr2int['\t']] = 1.0
        
        # Initialize beam for each batch item
        beams = []
        for b in range(batch_size):
            beam = [(0.0, [start_token], hidden_cell if isinstance(hidden_cell, torch.Tensor) 
                    else (hidden_cell[0][:, b:b+1, :], hidden_cell[1][:, b:b+1, :]))]
            beams.append(beam)
        
        for step in range(max_length):
            new_beams = []
            
            for b in range(batch_size):
                candidates = []
                
                for score, sequence, state in beams[b]:
                    if len(sequence) > 0:
                        last_token = sequence[-1]
                        
                        # Get decoder output
                        if self.atten == False:
                            prediction, new_state = self.decoder(
                                last_token.unsqueeze(0), state, encoder_output[:, b:b+1, :])
                        else:
                            prediction, new_state, _ = self.decoder(
                                last_token.unsqueeze(0), state, encoder_output[:, b:b+1, :])
                        
                        # Get top k predictions
                        log_probs = F.log_softmax(prediction, dim=-1)
                        top_probs, top_indices = torch.topk(log_probs, beam_width)
                        
                        for i in range(beam_width):
                            token_idx = top_indices[0, i].item()
                            token_prob = top_probs[0, i].item()
                            
                            # Create one-hot token
                            new_token = torch.zeros(1, self.num_decoder_tokens).to(self.device)
                            new_token[0, token_idx] = 1.0
                            
                            new_sequence = sequence + [new_token]
                            new_score = score + token_prob
                            
                            # Stop if end token
                            if token_idx == self.target_chr2int['\n']:
                                candidates.append((new_score, new_sequence, new_state))
                            else:
                                candidates.append((new_score, new_sequence, new_state))
                
                # Keep top beam_width candidates
                candidates.sort(key=lambda x: x[0], reverse=True)
                new_beams.append(candidates[:beam_width])
            
            beams = new_beams
        
        # Return best sequence for each batch item
        results = []
        for b in range(batch_size):
            best_beam = max(beams[b], key=lambda x: x[0])
            sequence_tokens = [token.argmax().item() for token in best_beam[1][1:]]  # Skip start token
            results.append(sequence_tokens)
        
        return results
