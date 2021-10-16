import torch.nn as nn
import logging
import hydra
log = logging.getLogger(__name__)
from src.model.module.ember.embedding import Embeder
import torch
from supar.modules import MLP, Biaffine


def identity(x):
    return x


class UniLSTMDec(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout_rate):
        super(UniLSTMDec, self).__init__()

        self._rnn = nn.LSTM(input_dim, output_dim, batch_first=True)
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden, hx=None, training=False):
        drop_h = self._dropout(hidden)
        if len(hidden.size()) == 2:
            drop_h = drop_h.unsqueeze(1)
        if hx is None:
            rnn_h, (nxt_s, nxt_c) = self._rnn(drop_h)
        else:
            state, cell = hx
            assert len(state.size()) == 2
            rnn_h, (nxt_s, nxt_c) = self._rnn(drop_h, (state.unsqueeze(0), cell.unsqueeze(0)))

        if not training:
            rnn_h = rnn_h.squeeze(1)

        return rnn_h, (nxt_s.squeeze(0), nxt_c.squeeze(0))

# (prev span, prev label, remain
class PointerNet(nn.Module):
    def __init__(self, conf, fields):
        super(PointerNet, self).__init__()
        self.conf = conf
        self.fields = fields
        self.vocab = self.fields.get_vocab('chart')

        self.metric = hydra.utils.instantiate(conf.metric.target, conf.metric, fields=fields, _recursive_=False)

        self.embeder =  Embeder(fields=self.fields, conf=self.conf.embeder)

        # self.decoder_linear = MLP(4 * self.conf.n_lstm_hidden + self.conf.label_emb_size, self.embeder.get_output_dim(), 0.33)
        self.decoder_linear = identity

        input_size = 0
        output_size = 2 * self.conf.n_lstm_hidden if self.conf.encoder_type == 'LSTM' else 2 * self.embeder.get_output_dim()

        if self.conf.use_prev_span:
            input_size +=  self.conf.input_span_size

        if self.conf.use_prev_label:
            input_size +=  self.conf.label_emb_size

        if self.conf.use_remain_span:
            input_size += self.conf.input_span_size

        assert input_size > 0
        self.decoder_input_size = input_size
        self.output_size = self.conf.n_lstm_hidden

        if self.conf.encoder_type == 'LSTM':
            from src.model.module.encoder.lstm_encoder import LSTMencoder
            self.encoder = LSTMencoder(self.conf.lstm_encoder, input_dim=self.embeder.get_output_dim())

        elif self.conf.encoder_type == 'self_attentive':
            from src.model.module.encoder.self_attentive import SelfAttentiveEncoder
            self.encoder = SelfAttentiveEncoder(self.conf.self_attentive_encoder, input_dim=self.embeder.get_output_dim())

        self.decoder = UniLSTMDec(input_dim= self.decoder_input_size, output_dim=self.conf.n_lstm_hidden, dropout_rate=self.conf.lstm_dropout)

        if self.conf.use_hx:
            self.hx_dense = MLP(2 * self.conf.n_lstm_hidden, self.conf.n_lstm_hidden)

        self.label_embedding = nn.Parameter(torch.rand(fields.get_vocab_size('chart'), conf.label_emb_size))

        if self.conf.use_remain_span:
            self.mlp_remain_span = MLP(n_in=output_size, n_out=self.conf.input_span_size , dropout=conf.lstm_dropout)

        if self.conf.use_prev_span:
            self.mlp_prev_span = MLP(output_size, self.conf.input_span_size, dropout=conf.lstm_dropout)

        additional_size = 0
        if self.conf.use_focus:
            additional_size = output_size


        self.mlp_src = MLP(n_in=output_size, n_out=self.conf.biaffine_size, dropout=conf.lstm_dropout)
        self.mlp_dec = MLP(n_in=self.conf.n_lstm_hidden + additional_size, n_out=self.conf.biaffine_size, dropout=conf.lstm_dropout)
        self.biaffine =  Biaffine(n_in=self.conf.biaffine_size, n_out=1,  bias_x=True, bias_y=False)

        self.label_projector = nn.Sequential(
        nn.Linear(output_size+self.conf.n_lstm_hidden, self.conf.label_emb_size),
        nn.LayerNorm(self.conf.label_emb_size),
        nn.ReLU(),
        nn.Linear(self.conf.label_emb_size, self.conf.label_emb_size),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.start_emb = nn.Parameter(torch.randn(1, output_size))
        self.start_label = nn.Parameter(torch.randn(1, self.conf.label_emb_size))


    def forward(self, ctx):
        # embeder
        self.embeder(ctx)

        if self.conf.encoder_type == 'LSTM':
            self.encoder(ctx)
            # fencepost representations for boundaries.
            output = ctx['encoded_emb']
            x_f, x_b = output.chunk(2, -1)
            repr =  torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
            hx = None
            return repr, hx

        else:
            self.encoder(ctx)
            return ctx['fencepost'], None

    def get_loss(self, x, y):
        ctx = {**x, **y}
        repr,  hx = self.forward(ctx)

        repr_biaffine = self.mlp_src(repr)

        seq_len = ctx['seq_len']
        batch_size = repr.shape[0]
        decoder_input_emb = []

        span_start = ctx['span_start']
        span_end = ctx['span_end']
        label = ctx['chart']

        end_repr = repr.gather(1, span_end.unsqueeze(-1).expand(*span_end.shape, repr.shape[-1]))
        start_repr = repr.gather(1, span_start.unsqueeze(-1).expand(*span_start.shape, repr.shape[-1]))
        gold_repr = end_repr - start_repr

        if self.conf.use_prev_span:
            # (batch, action_len+1, size)
            # first input is the especial start symbol.
            decoder_input_emb.append(
                self.mlp_prev_span(
                torch.cat([ self.start_emb.unsqueeze(0).expand(batch_size, 1, repr.shape[-1]), gold_repr[:, :-1]], dim=1)
                )
            )

        if self.conf.use_prev_label:
            assert self.conf.use_prev_span
            prev_label =  label[:, :-1]
            decoder_input_emb.append(
                torch.cat(
                    [self.start_label.unsqueeze(0).expand(batch_size, 1, self.conf.label_emb_size), self.label_embedding[prev_label]]
                , 1)
            )

        if self.conf.use_remain_span or self.conf.use_focus:
            remain_spans_start = torch.cat(
                [torch.zeros(batch_size, 1, device=span_end.device, dtype=torch.long), span_end[:, :-1]], dim=-1)
            remain_start_repr = repr.gather(1, remain_spans_start.unsqueeze(-1).expand(*span_start.shape, repr.shape[-1]))

        if self.conf.use_remain_span:
            # the first input is the entire span. so we need to pad 0.
            remain_spans_end = seq_len
            remain_end_repr = repr.gather(1, remain_spans_end.unsqueeze(-1).unsqueeze(-1).expand(*span_start.shape , repr.shape[-1]))
            decoder_input_emb.append(
                self.mlp_remain_span(
                remain_end_repr - remain_start_repr
                )
            )
            del remain_spans_start, remain_spans_end

        action_len = ctx['action_length']
        decoder_input = torch.cat(decoder_input_emb, dim=-1)
        action_mask = torch.arange(action_len.max(), device=decoder_input.device)[None, :] < action_len[:, None]

        output, _ = self.decoder(decoder_input,  hx, training=True)
        del decoder_input

        action_gold = ctx['action_golds']
        # batch_size * action_length * sen_len

        if self.conf.use_focus:
            output_biaffine = self.mlp_dec(
                torch.cat([
                    output, remain_start_repr
                ], dim=-1)
            )
        else:
            output_biaffine = self.mlp_dec(output)

        logits = self.biaffine.forward_blx_bay_2_bal(repr_biaffine, output_biaffine)

        # masked out invalid positions.
        logits = logits.masked_fill_(torch.arange(seq_len.max()+1, device=repr.device)[None, None, :] > seq_len[:, None, None], float('-inf'))

        # normalization
        logits, action_gold = logits[action_mask], action_gold[action_mask]
        pointer_loss = self.criterion(logits, action_gold)

        label_context = torch.cat([gold_repr, output], dim=-1)
        label_logits =  torch.matmul(self.label_projector(label_context), self.label_embedding.transpose(-1, -2))
        label_logits, label_gold = label_logits[action_mask], label[action_mask]
        label_loss = self.criterion(label_logits, label_gold)

        return (pointer_loss + label_loss)



    def decode(self, x, y):

        ctx = {**x, **y}

        repr, hx = self.forward(ctx)
        repr_biaffine = self.mlp_src(repr)

        batch_size = repr.shape[0]
        seq_len = ctx['seq_len']
        decoder_input_emb = []

        if self.conf.use_prev_span:
            decoder_input_emb.append(
               self.mlp_prev_span(self.start_emb).expand(batch_size, 500)
            )

        if self.conf.use_prev_label:
            decoder_input_emb.append(self.start_label.expand(batch_size, self.conf.label_emb_size))

        if self.conf.use_remain_span:
            decoder_input_emb.append(
                self.mlp_remain_span(
                repr.gather(1, seq_len.view(batch_size, 1, 1).expand(batch_size, 1, repr.shape[-1])).squeeze(1) - repr[:, 0]
                )
            )

        masks = torch.ones(size=(batch_size, seq_len.max() + 1), dtype=torch.bool, device=repr.device)

        # mask out invalid positions
        sent_masks = (torch.arange(seq_len.max() + 1, device=repr.device)[None, :] <= seq_len[:, None])
        masks = masks & sent_masks

        # [batch, num_hyp, sen_len]
        masks = masks.unsqueeze(1)
        previous_start_position = torch.zeros(batch_size, 1, dtype=torch.long, device=repr.device)
        start_position = torch.zeros(batch_size, 1, dtype=torch.long, device=repr.device)

        max_steps = 2*seq_len.max() - 1

        # (batch_size, num_hyp, max_steps) maintain the partial parse results.
        decoded_start_positions = start_position.new_zeros(batch_size, 1, max_steps)
        decoded_end_positions = start_position.new_zeros(batch_size, 1, max_steps)
        decoded_labels = start_position.new_zeros(batch_size, 1, max_steps)

        mask_stop = masks.new_zeros(batch_size, 1, )
        num_hyp = 1

        beam = self.conf.beam_size
        hypothesis_scores = repr.new_zeros((batch_size, 1))
        num_bd = seq_len.max() + 1



        for step in range(max_steps):

            dec_input = self.decoder_linear(torch.cat(decoder_input_emb, dim=-1).view(batch_size * num_hyp, -1))
            out, hx = self.decoder(dec_input, hx)
            dec_dim = out.size(1)
            out = out.view(batch_size, num_hyp, dec_dim)

            if self.conf.use_focus:
                out_biaffine = self.mlp_dec(
                    torch.cat(
                        [out,  repr.gather(1, start_position.unsqueeze(-1).expand(*start_position.shape, repr.shape[-1]))], dim=-1
                    )
                )
            else:
                out_biaffine = self.mlp_dec(out)

            # [batch, num_hyp, sent_len]
            current_masks = masks.clone()

            # mask out start position.
            current_masks.scatter_(2, previous_start_position.unsqueeze(-1), False)

            # repr: [batch, num_bd, src_dim],  out: [batch , num_hyp, dec_dim]
            pointer_scores = self.biaffine.forward_blx_bay_2_bal(repr_biaffine, out_biaffine)

            # [batch, num_hyp, dec_dim]
            # mask out invalid positions
            pointer_scores = pointer_scores.view(batch_size, num_hyp, pointer_scores.shape[-1]).masked_fill_(~sent_masks.unsqueeze(1), float('-inf'))

            # normalize之后，再把不满足parsing限制的位置的score set成-inf. (mask out all invalid positions)
            pointer_scores = pointer_scores.log_softmax(-1).masked_fill_(~current_masks, float('-inf'))

            # 如果这个beam已经stop了的话, 那么current_mask肯定全部的position都是mask=True的状态.
            # 那么pointer_score全部都是-inf. 我们把指向开头的pointer的score设成0, 这样的话这个beam就会维持原来的分数, 而且不会新增多的相同的beam.
            # TODO: check this.  seems correct. Date: 21/09/19
            mask_stop_tmp = mask_stop.unsqueeze(-1).repeat(1, 1, num_bd)
            mask_stop_tmp[..., 1:] = False
            pointer_scores = pointer_scores.masked_fill_(mask_stop_tmp, 0)

            # beam search and backtracking.
            hypothesis_scores = hypothesis_scores.unsqueeze(2) + pointer_scores
            hypothesis_scores, hyp_index = torch.sort(hypothesis_scores.view(batch_size, -1), dim=1, descending=True)
            prev_num_hyp = num_hyp

            # 所有的hyp = 所有能够被point到的位置 + 已经停止了的hyp
            num_hyp = ((current_masks).view(batch_size, -1).sum(dim=1) + mask_stop.sum(-1)) .max().clamp(max=beam).item()
            hypothesis_scores = hypothesis_scores[:, :num_hyp]
            hyp_index = hyp_index[:, :num_hyp]
            base_index = (hyp_index // num_bd)
            selected_index = hyp_index % num_bd
            focus_index = start_position.gather(1, base_index)

            # --------------- 这是用来维持previously parsed的代码部分 ----------
            #  maintain all previously parsed parts
            base_index_expand = base_index.unsqueeze(-1).expand(*base_index.shape, max_steps)
            decoded_start_positions = decoded_start_positions.gather(1, base_index_expand)
            decoded_end_positions = decoded_end_positions.gather(1, base_index_expand)
            decoded_labels = decoded_labels.gather(1, base_index_expand)

            masks = masks.gather(1, base_index.unsqueeze(-1).expand(*base_index.shape, num_bd))
            out = out.gather(1, base_index.unsqueeze(-1).expand(*base_index.shape, out.shape[-1]))

            # [batch, num_hyp,]
            start = torch.min(focus_index, selected_index)
            end = torch.max(focus_index, selected_index)

            # [batch, num_hyp, max_len]
            decoded_start_positions[:, :, step] = start
            decoded_end_positions[:, :, step] = end
            # --------------- 这是用来维持previously parsed的代码部分 END----------

            start_position = start
            end_position = end

            # [batch, num_hyp, num_bd]
            masks2 =  (torch.arange(seq_len.max()+1, device=masks.device)[None, None, :] <= end_position[..., None]) & (torch.arange(seq_len.max()+1, device=masks.device)[None, None, :]  >= start_position[..., None])

            masks = masks & ~masks2
            # [b, hypo, dec_dim]
            prev_span_repr = repr.gather(1, end.unsqueeze(-1).expand(*end.shape, repr.shape[-1])) - repr.gather(1, start.unsqueeze(-1).expand(*end.shape, repr.shape[-1]))
            label_context = torch.cat([prev_span_repr, out], dim=-1)

            # [b, hypo]
            label_preds = torch.matmul(self.label_projector(label_context), self.label_embedding.transpose(-1, -2)).argmax(-1)
            decoded_labels[:, :, step] = label_preds

            mask_stop = (masks.sum(-1) == 0)

            # 所有的beam都停止啦, 提前退出.
            # all beams are finished, early exit.
            if mask_stop.all():
                break

            masks.scatter_(2, start_position.unsqueeze(-1), True)
            previous_start_position = start_position
            start_position = end_position

            decoder_input_emb = []

            if self.conf.use_prev_span:
                decoder_input_emb.append(self.mlp_prev_span(prev_span_repr))

            if self.conf.use_prev_label:
                decoder_input_emb.append(self.label_embedding[label_preds])

            if self.conf.use_remain_span:
                decoder_input_emb.append(
                    self.mlp_remain_span(
                        repr.gather(1, seq_len.unsqueeze(-1).unsqueeze(-1).expand(*end.shape, repr.shape[-1])) -  repr.gather(1, end.unsqueeze(-1).expand(*end.shape, repr.shape[-1]))
                    )
                )

            #更换LSTM的hidden state的部分.
            # change the hidden state of LSTM for each beam.
            batch_index = seq_len.new_tensor(range(batch_size)).view(batch_size, 1)
            hx_index = (base_index + batch_index * prev_num_hyp).view(batch_size * num_hyp)
            hx, cx = hx
            hx = hx[hx_index]
            cx = cx[hx_index]
            hx = (hx, cx)

        # decode部分
        # 取score最高的beam (obtaining the highest-scoring beam)
        decoded_start_positions = decoded_start_positions[:, 0, ].cpu().numpy()
        decoded_end_positions = decoded_end_positions[:, 0, ].cpu().numpy()
        decoded_labels = decoded_labels[:, 0, ].cpu().numpy()
        results = []
        # recover constituent spans.
        for i in range(batch_size):
            result = []
            j = 0
            indicator = [True for _ in range(int(seq_len[i]))]
            while not (decoded_start_positions[i][j] == 0 and decoded_end_positions[i][j] == int(seq_len[i])):
                result.append( (decoded_start_positions[i][j], decoded_end_positions[i][j], decoded_labels[i][j]))
                if decoded_start_positions[i][j] + 1 == decoded_end_positions[i][j]:
                    indicator[decoded_start_positions[i][j]] = False
                j+=1

            result.append((decoded_start_positions[i][j], decoded_end_positions[i][j], decoded_labels[i][j]))
            for idx, v in enumerate(indicator):
                if v:
                    result.append((idx, idx+1, -1))
            results.append(result)
        ctx['chart_preds'] = results
        return ctx
