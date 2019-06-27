# coding=utf-8
import torch

DATA_ROOT = "./data"
ESP = 1e-5

# define different dialogue scenarios in different tasks
AVAILABLE_INTENT_1 = {"pre_sales": ["qa"],
                      "in_sales": ["payment", "discountURL", "expressTime", "expressName"]}

AVAILABLE_INTENT_2 = {"pre_sales": ["qa", "confirm"],
                      "in_sales": ["payment", "discountURL", "expressTime", "expressName"]}

AVAILABLE_INTENT_3 = {"pre_sales": ["qa", "confirm", "compare"],
                      "in_sales": ["payment", "discountURL", "expressTime", "expressName"]}

AVAILABLE_INTENT_4 = {"pre_sales": ["qa", "confirm", "compare"],
                      "in_sales": ["payment", "discountURL", "expressTime", "expressName"],
                      "after_sales": ["expressInfo", "invoice", "exchange", "exchange_exchange", "consult", "refund",
                                      "consult_refund"]}

AVAILABLE_INTENT_5 = {"pre_sales": ["qa", "confirm", "compare"],
                      "in_sales": ["payment", "discountURL", "expressTime", "expressName"],
                      "after_sales": ["expressInfo", "invoice", "exchange", "exchange_exchange", "consult", "refund",
                                      "consult_refund"],
                      "sentiment": ["positive", "negative"]}
TASKS = {"task_1": AVAILABLE_INTENT_1, "task_2": AVAILABLE_INTENT_2, "task_3": AVAILABLE_INTENT_3,
         "task_4": AVAILABLE_INTENT_4, "task_5": AVAILABLE_INTENT_5}


class RunConfig(object):
    #########################################
    # Change the parameters here!
    #########################################
    trained_model = None
    # "task_1" , "task_2", ..., "task_5"
    coming_task = "task_1"
    # "deploy" means deploying system from scratch and online updating,
    # "test" means freezing online learning module
    system_mode = "deploy"
    model_save_path = "checkpoints/task_1_deploy_from_scratch_model.pkl"
    debug_path = "debug/task_1_deploy_from_scratch_debug.pkl"

    random_seed = 42
    lr = 0.001
    batch_size = 32
    max_clip = 40.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word_emb_size = 32

    # sentence encoding method
    sent_encode_method = "rnn"
    emb_sum = False
    sent_rnn_type = "gru"
    sent_rnn_hidden_size = 32
    sent_rnn_layers = 1
    sent_rnn_dropout = 0
    sent_rnn_bidirectional = True

    sent_self_attn = True
    sent_self_attn_hidden = 32
    sent_self_attn_head = 1

    # context encoding method
    ctx_encode_method = "HierarchalRNN"
    attn_method = "general"
    memory_size = 50
    max_hops = 3
    memory_nonlinear = "iden"
    ctx_rnn_type = "gru"
    ctx_rnn_hidden_size = 32
    ctx_rnn_layers = 1
    ctx_rnn_dropout = 0
    ctx_rnn_bidirectional = False

    ctx_self_attn = False
    ctx_self_attn_hidden = 32
    ctx_self_attn_head = 1

    latent_size = 20
    prior_sample = 50
    posterior_sample = 50
    max_jsd = 0.3
    min_prob = 0.3
    full_kl_step = 10000

    if sent_encode_method == "rnn":
        if sent_self_attn is True:
            sent_emb_size = sent_rnn_hidden_size * 2 if sent_rnn_bidirectional else sent_rnn_hidden_size
        else:
            sent_emb_size = sent_rnn_hidden_size * sent_rnn_layers * 2 if sent_rnn_bidirectional \
                else sent_rnn_hidden_size * sent_rnn_layers
    elif sent_encode_method == "bow":
        sent_emb_size = word_emb_size

    if ctx_encode_method == "MemoryNetwork" or ctx_encode_method == "HierarchalSelfAttn":
        ctx_emb_size = sent_emb_size
    elif ctx_encode_method == "HierarchalRNN":
        if ctx_self_attn is True:
            ctx_emb_size = ctx_rnn_hidden_size * 2 if ctx_rnn_bidirectional else ctx_rnn_hidden_size
        else:
            ctx_emb_size = ctx_rnn_hidden_size * ctx_rnn_layers * 2 if ctx_rnn_bidirectional \
                else ctx_rnn_hidden_size * ctx_rnn_layers
