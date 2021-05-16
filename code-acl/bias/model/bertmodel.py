from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
# from transformers import DistilBertTokenizerFast, DistilBertPreTrainedModel, DistilBertModel
from parameters import BERT_MODEL_PATH, CLAIM_ONLY, CLAIM_AND_EVIDENCE, EVIDENCE_ONLY, DEVICE
from torch.nn import functional as F
import torch.nn as nn
import torch

class MyBertModel(BertPreTrainedModel):
    def __init__(self, config, labelnum, maxlen=200, input_type=CLAIM_ONLY):

        super(MyBertModel, self).__init__(config)
        self.input_type = input_type
        self.maxlen = maxlen

        self.tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_PATH)

        self.bert = BertModel(config)

        if input_type == CLAIM_ONLY or input_type == EVIDENCE_ONLY:
            self.predictor = nn.Linear(768, labelnum)
        elif input_type == CLAIM_AND_EVIDENCE:
            self.predictor = nn.Linear(768 * 2, labelnum)

        if self.input_type != CLAIM_ONLY:
            self.attn_score = nn.Linear(768, 1)
            self.softmax = nn.Softmax(dim=1)

        self.init_weights()

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
        tmp = self.tokenizer(claims, return_tensors='pt', padding=True, truncation=True, max_length=self.maxlen)
        input_ids = tmp["input_ids"].to(DEVICE)
        attention_mask = tmp["attention_mask"].to(DEVICE)

        return input_ids, attention_mask

    def encode_snippets(self, snippets):
        concat_snippets = [item for sublist in snippets for item in sublist.tolist()]
        tmp = self.tokenizer(concat_snippets, return_tensors='pt', padding=True, truncation=True, max_length=self.maxlen)
        input_ids = tmp["input_ids"].to(DEVICE)
        token_type_ids = tmp["token_type_ids"].to(DEVICE)
        attention_mask = tmp["attention_mask"].to(DEVICE)
        return input_ids, token_type_ids, attention_mask

    def encode_snippets_with_claims(self, snippets, claims):
        concat_claims = []
        for claim in claims:
            concat_claims += [claim]*10

        concat_snippets = [item for sublist in snippets for item in sublist.tolist()]

        tmp = self.tokenizer(concat_claims, concat_snippets, return_tensors='pt', padding=True, truncation=True, max_length=self.maxlen)

        input_ids = tmp["input_ids"].to(DEVICE)
        token_type_ids = tmp["token_type_ids"].to(DEVICE)
        attention_mask = tmp["attention_mask"].to(DEVICE)

        return input_ids, token_type_ids, attention_mask

    def predict_claim(self, claims):
        claim_input_ids, claim_attn_masks = self.encode_claims(claims)
        cls = self.bert(claim_input_ids, attention_mask=claim_attn_masks, )[0][:,0,:]
        return self.predictor(cls)

    def predict_evidence(self, snippets):
        snippet_input_ids, snippet_token_type_ids, snippet_attention_mask = self.encode_snippets(snippets)
        snippet_cls = self.bert(snippet_input_ids, token_type_ids=snippet_token_type_ids, attention_mask=snippet_attention_mask)[0][:,0,:]
        snippet_cls = snippet_cls.view(len(snippets), 10, 768)

        tmp = self.attn_score(snippet_cls)
        attn_weights = self.softmax(tmp)
        snippet_cls = snippet_cls * attn_weights
        snippet_cls = torch.sum(snippet_cls, dim=1)

        return self.predictor(snippet_cls)

    def predict_claim_evidence(self, claims, snippets):
        claim_input_ids, claim_attn_masks = self.encode_claims(claims)
        claim_cls = self.bert(claim_input_ids, attention_mask=claim_attn_masks)[0][:, 0, :]

        snippet_input_ids, snippet_token_type_ids, snippet_attention_mask = self.encode_snippets_with_claims(snippets, claims)
        snippet_cls = self.bert(snippet_input_ids, token_type_ids=snippet_token_type_ids, attention_mask=snippet_attention_mask)[0][:,0,:]
        snippet_cls = snippet_cls.view(len(claims), 10, 768)

        tmp = self.attn_score(snippet_cls)
        attn_weights = self.softmax(tmp)
        snippet_cls *= attn_weights
        snippet_cls = torch.sum(snippet_cls, dim=1)

        claim_snippet_cls = torch.cat((claim_cls, snippet_cls), dim=-1)

        return self.predictor(claim_snippet_cls)
