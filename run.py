# coding = utf-8
import torch
import time

from cvae import ContinuousAgent, ContinuousVAE
from data_utils import DataUtils
from config import RunConfig as Config

if __name__ == "__main__":
    config = Config()
    api = DataUtils(config.ctx_encode_method)
    api.load_vocab()
    api.load_candidates()
    api.load_dialog(config.coming_task, config.system_mode)
    api.build_pad_config(config.memory_size)

    model = ContinuousVAE(config, api)

    if config.trained_model is not None:
        print("Using trained model in {}".format(config.trained_model))
        model.load_state_dict(torch.load(config.trained_model))

    agent = ContinuousAgent(config, model, api)

    t1 = time.time()
    agent.main()
    t2 = time.time()
    print("cost time: {} seconds.".format(t2 - t1))
