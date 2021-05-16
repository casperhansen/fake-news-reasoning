from parameters import CLAIM_ONLY, CLAIM_AND_EVIDENCE, EVIDENCE_ONLY, DEVICE
from torch.nn import functional as F
import torch.nn as nn
import torch
from utils.utils import clean_str
from bias.parameters import MASK_TOKEN, UNK_TOKEN

class LSTMModel(nn.Module):
    def __init__(self, hidden_dim, lstm_layer, dropout, labelnum, word2idx, glove_embedding_matrix, maxlen=200, input_type=CLAIM_ONLY):

        super(LSTMModel, self).__init__()
        self.input_type = input_type
        self.maxlen = maxlen
        self.word2idx = word2idx
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(glove_embedding_matrix, dtype=torch.float32))
        self.hidden_dim = hidden_dim
        self.lstm_layer = lstm_layer

        self.claim_lstm = nn.LSTM(input_size=300,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=float(dropout),
                            bidirectional=True)

        self.snippet_lstm = nn.LSTM(input_size=300,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout=float(dropout),
                            bidirectional=True)

        self.attn_score_claim = nn.Linear(hidden_dim * 2, 1)
        self.attn_score_snippet = nn.Linear(hidden_dim * 2, 1)
        self.attn_score_snippet_level = nn.Linear(hidden_dim * 2, 1) # used for summarizing 10 snippets into one tensor

        self.attn_score_claim_snippet = nn.Linear(hidden_dim * 2 * 4, 1)

        if input_type != CLAIM_AND_EVIDENCE:
            self.score = nn.Linear(hidden_dim * 2, labelnum)
        else:
            self.score = nn.Linear(hidden_dim * 2 * 4, labelnum)

        self.softmax = nn.Softmax(dim=1)

    def tokenizer(self, list_of_sentences, predefined_max_len):
        out = []
        for sent in list_of_sentences:
            out.append([self.word2idx[v] if v in self.word2idx else UNK_TOKEN for v in clean_str(sent).split(" ")])

        max_len = max([len(v) for v in out])
        max_len = min(max_len, predefined_max_len)

        for i in range(len(out)):
            out_len = len(out[i])
            if out_len < max_len:
                out[i] += [MASK_TOKEN for _ in range(max_len - out_len)]
            else:
                out[i] = out[i][:max_len]

        return torch.tensor(out)

    def forward(self, claims, snippets):
        if self.input_type == CLAIM_ONLY:
            return self.predict_claim(claims)
        elif self.input_type == EVIDENCE_ONLY:
            return self.predict_evidence(snippets)
        elif self.input_type == CLAIM_AND_EVIDENCE:
            return self.predict_claim_evidence(claims, snippets)
        else:
            raise Exception("Unknown type", self.input_type)

    def encode_claims(self, claims):
        tmp = self.tokenizer(claims, self.maxlen).to(DEVICE)
        tmp = self.embedding(tmp)
        return tmp

    def encode_snippets(self, snippets):
        tmp = [item for sublist in snippets for item in sublist.tolist()]
        tmp = self.tokenizer(tmp, self.maxlen).to(DEVICE)
        tmp = self.embedding(tmp)
        return tmp

    def attn(self, three_dim_tensor, attn_score): # batch * tokens * dim
        tmp = attn_score(three_dim_tensor)
        attn_weights = self.softmax(tmp)
        three_dim_tensor = three_dim_tensor * attn_weights
        three_dim_tensor = torch.sum(three_dim_tensor, dim=1)
        return three_dim_tensor, attn_weights.squeeze(-1)

    def predict_claim(self, claims):
        claims = self.encode_claims(claims)
        lstm_out, _ = self.claim_lstm(claims)
        lstm_out, _ = self.attn(lstm_out, self.attn_score_claim)
        score = self.score(lstm_out)
        return score

    def predict_evidence(self, snippets):
        batchsize = len(snippets)
        snippets = self.encode_snippets(snippets)
        lstm_out, _ = self.claim_lstm(snippets)
        lstm_out, _ = self.attn(lstm_out, self.attn_score_snippet)
        lstm_out = lstm_out.view(batchsize, 10, self.hidden_dim * 2)
        lstm_out, snippet_attn = self.attn(lstm_out, self.attn_score_snippet_level)
        score = self.score(lstm_out)

        return score

    def predict_claim_evidence(self, claims, snippets):
        claims = self.encode_claims(claims)
        claims_lstm_out, _ = self.claim_lstm(claims)
        claims_lstm_out, _ = self.attn(claims_lstm_out, self.attn_score_claim)

        batchsize = len(snippets)
        snippets = self.encode_snippets(snippets)
        snippets_lstm_out, _ = self.claim_lstm(snippets)
        snippets_lstm_out, _ = self.attn(snippets_lstm_out, self.attn_score_snippet)
        snippets_lstm_out = snippets_lstm_out.view(batchsize, 10, self.hidden_dim * 2)

        claims_lstm_out = claims_lstm_out.unsqueeze(1).repeat(1, 10, 1)
        # combined representation
        combined_lstm_out = (claims_lstm_out, snippets_lstm_out, claims_lstm_out - snippets_lstm_out, claims_lstm_out * snippets_lstm_out) # [a, b, a - b, a * b]
        combined_lstm_out = torch.cat(combined_lstm_out, dim=-1)
        combined_lstm_out, snippet_attn = self.attn(combined_lstm_out, self.attn_score_claim_snippet)

        print(combined_lstm_out.size())
        exit()

        score = self.score(combined_lstm_out)
        return score
