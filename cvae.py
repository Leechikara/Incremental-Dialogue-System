# coding = utf-8
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
import pickle
from sklearn import metrics
import scipy.special as special
from config import ESP, DATA_ROOT
from data_utils import batch_iter
from nn_utils import Attn, bow_sentence, bow_sentence_self_attn, rnn_seq, rnn_seq_self_attn, RnnV, SelfAttn


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = - 0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                            - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                            - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
    return kld


def sample_gaussian(mu, logvar, n):
    # return (batch, n, z_dim)
    eps_shape = list(mu.shape)
    eps_shape.insert(1, n)
    mu_temp = torch.unsqueeze(mu, 1)
    logvar_temp = torch.unsqueeze(logvar, 1)
    epsilon = torch.randn(eps_shape, device=mu.device)
    std = torch.exp(0.5 * logvar_temp)
    z = mu_temp + std * epsilon
    return z


def jsd(probs, probs_mean):
    """
    :param probs: (batch_size, time, candidate_size)
    :param probs_mean: (batch_size, 1, candidate_size)
    :return: (batch_size, )
    """
    time = probs.shape[1]
    probs_mean = np.repeat(probs_mean, time, axis=1)
    return ((special.kl_div(probs, probs_mean).sum(2) + special.kl_div(probs_mean, probs).sum(2)) / 2).sum(1) / time


class ContinuousVAE(nn.Module):
    def __init__(self, config, api):
        super(ContinuousVAE, self).__init__()
        torch.manual_seed(config.random_seed)

        self.api = api
        self.config = config

        self.embedding = nn.Embedding(self.api.vocab_size, self.config.word_emb_size, padding_idx=0)

        if self.config.sent_encode_method == "rnn":
            self.sent_rnn = RnnV(self.config.word_emb_size, self.config.sent_rnn_hidden_size,
                                 self.config.sent_rnn_type, self.config.sent_rnn_layers,
                                 dropout=self.config.sent_rnn_dropout,
                                 bidirectional=self.config.sent_rnn_bidirectional)
        if self.config.sent_self_attn is True:
            self.sent_self_attn_layer = SelfAttn(self.config.sent_emb_size,
                                                 self.config.sent_self_attn_hidden,
                                                 self.config.sent_self_attn_head)

        if self.config.ctx_encode_method == "MemoryNetwork":
            self.attn_layer = Attn(self.config.attn_method, self.config.sent_emb_size, self.config.sent_emb_size)
            self.hops_map = nn.Linear(self.config.sent_emb_size, self.config.sent_emb_size)
            if self.config.memory_nonlinear.lower() == "tanh":
                self.memory_nonlinear = nn.Tanh()
            elif self.config.memory_nonlinear.lower() == "relu":
                self.memory_nonlinear = nn.ReLU()
            elif self.config.memory_nonlinear.lower() == "iden":
                self.memory_nonlinear = None
        elif self.config.ctx_encode_method == "HierarchalRNN":
            self.ctx_rnn = RnnV(self.config.sent_emb_size, self.config.ctx_rnn_hidden_size,
                                self.config.ctx_rnn_type, self.config.ctx_rnn_layers,
                                dropout=self.config.ctx_rnn_dropout,
                                bidirectional=self.config.ctx_rnn_bidirectional)
            if self.config.ctx_self_attn is True:
                self.ctx_self_attn_layer = SelfAttn(self.config.ctx_emb_size,
                                                    self.config.ctx_self_attn_hidden,
                                                    self.config.ctx_self_attn_head)
        elif self.config.ctx_encode_method == "HierarchalSelfAttn":
            self.ctx_self_attn_layer = SelfAttn(self.config.ctx_emb_size,
                                                self.config.ctx_self_attn_hidden,
                                                self.config.ctx_self_attn_head)

        cond_emb_size = self.config.ctx_emb_size
        response_emb_size = self.config.sent_emb_size
        recog_input_size = cond_emb_size + response_emb_size

        self.recogNet_mulogvar = nn.Sequential(
            nn.Linear(recog_input_size, max(50, self.config.latent_size * 2)),
            nn.Tanh(),
            nn.Linear(max(50, self.config.latent_size * 2), self.config.latent_size * 2)
        )

        self.priorNet_mulogvar = nn.Sequential(
            nn.Linear(cond_emb_size, max(50, self.config.latent_size * 2)),
            nn.Tanh(),
            nn.Linear(max(50, self.config.latent_size * 2), self.config.latent_size * 2)
        )

        self.fused_cond_z = nn.Linear(cond_emb_size + self.config.latent_size, self.config.sent_emb_size)
        self.drop = nn.Dropout(p=0.5)

        # Record all candidates in advance and current available index.
        # If human give new a response, we add its index to available_cand_index.
        # If deploy from scratch, we assume that only candidates response in task_1 are known to developers.
        self.available_cand_index = list()
        if self.config.system_mode in ["deploy"]:
            # deploy IDS from scratch
            with open(os.path.join(DATA_ROOT, "candidate", "task_1.txt")) as f:
                for line in f:
                    line = line.strip()
                    self.available_cand_index.append(api.candid2index[line])
        else:
            # test IDS
            with open(os.path.join(DATA_ROOT, "candidate", self.config.coming_task + ".txt")) as f:
                for line in f:
                    line = line.strip()
                    if api.candid2index[line] not in self.available_cand_index:
                        self.available_cand_index.append(api.candid2index[line])
        self.available_cand_index.sort()
        self.register_buffer("candidates", torch.from_numpy(api.vectorize_candidates()))

    def sent_encode(self, s):
        if self.config.sent_encode_method == "bow":
            if self.config.sent_self_attn is False:
                s_encode = bow_sentence(self.embedding(s), self.config.emb_sum)
            else:
                s_encode = bow_sentence_self_attn(self.embedding(s), self.sent_self_attn_layer)
        elif self.config.sent_encode_method == "rnn":
            if self.config.sent_self_attn is False:
                s_encode = rnn_seq(self.embedding(s), self.sent_rnn, self.config.sent_emb_size)
            else:
                s_encode = rnn_seq_self_attn(self.embedding(s), self.sent_rnn,
                                             self.sent_self_attn_layer, self.config.sent_emb_size)
        return s_encode

    def ctx_encode_m2n(self, contexts):
        stories, queries = contexts
        m = self.sent_encode(stories)
        q = self.sent_encode(queries)

        u = [q]

        for _ in range(self.config.max_hops):
            # attention over memory and read memory
            _, o_k = self.attn_layer(m, u[-1])

            # fuse read memory and previous hops
            u_k = self.hops_map(u[-1]) + o_k
            if self.memory_nonlinear is not None:
                u_k = self.memory_nonlinear(u_k)

            u.append(u_k)

        return u[-1]

    def ctx_encode_h_self_attn(self, contexts):
        stories, _ = contexts
        m = self.sent_encode(stories)
        return self.ctx_self_attn_layer(m)

    def ctx_encode_h_rnn(self, contexts):
        stories, _ = contexts
        m = self.sent_encode(stories)

        if self.config.ctx_self_attn is True:
            return rnn_seq_self_attn(m, self.ctx_rnn, self.ctx_self_attn_layer, self.config.ctx_emb_size)
        else:
            return rnn_seq(m, self.ctx_rnn, self.config.ctx_emb_size)

    def select_uncertain_points(self, logits):
        probs = F.softmax(logits, 2)
        probs = probs.cpu().detach().numpy()

        probs = probs + ESP
        probs_mean = np.mean(probs, axis=1, keepdims=True)

        js_distance = jsd(probs, probs_mean)
        js_selected = js_distance < self.config.max_jsd

        probs_mean = np.squeeze(probs_mean, axis=1)
        max_probs = np.max(probs_mean, axis=1)
        max_select = max_probs > self.config.min_prob

        selected = js_selected * max_select

        selected_responses = probs.mean(axis=1).argmax(axis=1)

        uncertain_index = list()
        certain_index = list()
        certain_response = list()
        for i, (selected_flag, selected_r) in enumerate(zip(selected, selected_responses)):
            if selected_flag:
                certain_index.append(i)
                certain_response.append(self.available_cand_index[selected_r])
            else:
                uncertain_index.append(i)

        return uncertain_index, certain_index, certain_response

    @staticmethod
    def evaluate(certain_index, certain_responses, feed_dict):
        feed_responses = np.array([feed_dict["responses"][i] for i in certain_index])
        certain_responses = np.array(certain_responses)
        acc = metrics.accuracy_score(feed_responses, certain_responses)
        return acc

    def tensor_wrapper(self, data):
        if isinstance(data, list):
            data = np.array(data)
        data = torch.from_numpy(data)
        return data.to(self.config.device)

    def forward(self, feed_dict):
        if self.config.ctx_encode_method == "MemoryNetwork":
            context_rep = self.ctx_encode_m2n(feed_dict["contexts"])
        elif self.config.ctx_encode_method == "HierarchalSelfAttn":
            context_rep = self.ctx_encode_h_self_attn(feed_dict["contexts"])
        elif self.config.ctx_encode_method == "HierarchalRNN":
            context_rep = self.ctx_encode_h_rnn(feed_dict["contexts"])
        cond_emb = context_rep

        prior_mulogvar = self.priorNet_mulogvar(cond_emb)
        prior_mu, prior_logvar = torch.chunk(prior_mulogvar, 2, 1)
        latent_prior = sample_gaussian(prior_mu, prior_logvar, self.config.prior_sample)

        cond_emb_temp = cond_emb.unsqueeze(1).expand(-1, self.config.prior_sample, -1)
        cond_z_embed_prior = self.fused_cond_z(torch.cat([cond_emb_temp, latent_prior], 2))
        candidates_rep = self.sent_encode(self.candidates)
        current_candidates_rep = candidates_rep[self.available_cand_index]
        logits = torch.matmul(cond_z_embed_prior, current_candidates_rep.t())
        uncertain_index, certain_index, certain_response = self.select_uncertain_points(logits)

        if len(certain_index) > 0:
            acc = self.evaluate(certain_index, certain_response, feed_dict)
        else:
            acc = None

        if self.config.system_mode == "test":
            return uncertain_index, certain_index, certain_response, acc

        if len(uncertain_index) > 0:
            # Simulate human in the loop and update the available response set
            uncertain_resp_index = [int(feed_dict["responses"][i]) for i in uncertain_index]
            self.available_cand_index = list(set(self.available_cand_index) | set(uncertain_resp_index))
            self.available_cand_index.sort()
            current_candidates_rep = candidates_rep[self.available_cand_index]

            uncertain_cond_emb = cond_emb[uncertain_index]
            uncertain_resp_emb = candidates_rep[uncertain_resp_index]

            recog_input = torch.cat([uncertain_cond_emb, uncertain_resp_emb], 1)

            posterior_mulogvar = self.recogNet_mulogvar(recog_input)
            posterior_mu, posterior_logvar = torch.chunk(posterior_mulogvar, 2, 1)
            latent_posterior = sample_gaussian(posterior_mu, posterior_logvar, self.config.posterior_sample)

            # loss
            uncertain_cond_emb_temp = uncertain_cond_emb.unsqueeze(1).expand(-1, self.config.posterior_sample, -1)
            cond_z_embed_posterior = self.fused_cond_z(
                torch.cat([self.drop(uncertain_cond_emb_temp), latent_posterior], 2))
            uncertain_logits = torch.matmul(cond_z_embed_posterior, current_candidates_rep.t()).contiguous()
            uncertain_logits = uncertain_logits.view(-1, uncertain_logits.size(2))

            target = list(map(lambda resp_index: self.available_cand_index.index(resp_index), uncertain_resp_index))
            target = torch.Tensor(target).to(uncertain_logits.device, dtype=torch.long)
            target = target.unsqueeze(1).expand(-1, self.config.posterior_sample).contiguous().view(-1)

            avg_rc_loss = F.cross_entropy(uncertain_logits, target)
            kld = gaussian_kld(posterior_mu, posterior_logvar,
                               prior_mu[uncertain_index], prior_logvar[uncertain_index])
            avg_kld = torch.mean(kld)
            kl_weights = min(feed_dict["step"] / self.config.full_kl_step, 1)
            elbo = avg_rc_loss + avg_kld * kl_weights
        else:
            elbo = None

        return elbo, uncertain_index, certain_index, certain_response, acc


class ContinuousAgent(object):
    def __init__(self, config, model, api):
        np.random.seed(config.random_seed + 1)
        self.config = config
        self.model = model.to(config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.api = api
        self.api.vectorize_data(self.api.data)

    def tensor_wrapper(self, data):
        if isinstance(data, list):
            data = np.array(data)
        data = torch.from_numpy(data)
        return data.to(self.config.device)

    def main(self):
        if self.config.system_mode == "deploy":
            with torch.set_grad_enabled(True):
                self.simulate_run()
        elif self.config.system_mode == "test":
            with torch.set_grad_enabled(False):
                self.test()

    def test(self):
        uncertain = list()
        certain = list()
        acc_in_certain = list()

        for step, (s, q, a, start) in enumerate(
                batch_iter(self.api.comingS, self.api.comingQ, self.api.comingA, self.config.batch_size)):
            feed_dict = {"contexts": (self.tensor_wrapper(s), self.tensor_wrapper(q)),
                         "responses": a, "step": step, "start": start}

            uncertain_index, certain_index, _, acc = self.model(feed_dict)
            uncertain.append(len(uncertain_index))
            certain.append(len(certain_index))
            acc_in_certain.append(acc)

        # Debug
        acc_num = 0
        for acc, certain_num in zip(acc_in_certain, certain):
            if acc is not None:
                acc_num += acc * certain_num

        print("In testing, we have {} points certain, "
              "{} points uncertain. "
              "In the certain points, {} points are right. "
              "The rate is {}.".format(sum(certain), sum(uncertain), acc_num, acc_num / sum(certain)))

    def simulate_run(self):
        log = defaultdict(list)

        for step, (s, q, a, start) in enumerate(
                batch_iter(self.api.comingS, self.api.comingQ, self.api.comingA, self.config.batch_size, shuffle=True)):
            self.optimizer.zero_grad()
            feed_dict = {"contexts": (self.tensor_wrapper(s), self.tensor_wrapper(q)),
                         "responses": a, "step": step, "start": start}

            loss, uncertain_index, certain_index, _, acc_in_certain = self.model(feed_dict)

            if loss is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_clip)
                self.optimizer.step()

            log["uncertain"].append(len(uncertain_index))
            log["certain"].append(len(certain_index))
            log["acc_in_certain"].append(acc_in_certain)
            log["loss"].append(loss.item())

        torch.save(self.model.state_dict(), self.config.model_save_path)
        pickle.dump(log, open(self.config.debug_path, "wb"))

        # Debug
        acc_num = 0
        for acc, certain_num in zip(log["acc_in_certain"], log["certain"]):
            if acc is not None:
                acc_num += acc * certain_num
        print("In deployment stage, we have {} points certain, "
              "{} points uncertain. "
              "In the certain points, {} points are right. "
              "The rate is {}.".format(sum(log["certain"]), sum(log["uncertain"]),
                                       acc_num,
                                       (acc_num / sum(log["certain"])) if sum(log["certain"]) > 0 else None))
