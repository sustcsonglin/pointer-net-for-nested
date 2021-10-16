import torch.nn as nn
import logging
import hydra
log = logging.getLogger(__name__)

class Parser(nn.Module):
    def __init__(self, conf, fields):
        super(Parser, self).__init__()
        self.conf = conf
        self.fields = fields
        self.embeder = hydra.utils.instantiate(conf.embeder.target, conf.embeder, fields=fields, _recursive_=False)
        self.encoder = hydra.utils.instantiate(conf.encoder.target, conf.encoder, input_dim= self.embeder.get_output_dim(), _recursive_=False)
        self.scorer = hydra.utils.instantiate(conf.scorer.target, conf.scorer, fields=fields, input_dim=self.encoder.get_output_dim(),  _recursive_=False)
        self.loss = hydra.utils.instantiate(conf.loss.target, conf.loss, _recursive_=False)
        self.metric = hydra.utils.instantiate(conf.metric.target, conf.metric, fields=fields, _recursive_=False)

        log.info(self.embeder)
        log.info(self.encoder)
        log.info(self.scorer)

    def forward(self, ctx):
        self.embeder(ctx)
        self.encoder(ctx)
        self.scorer(ctx)

    def get_loss(self, x, y):
        ctx = {**x, **y}
        self.forward(ctx)
        return self.loss.loss(ctx)

    def decode(self, x, y):
        ctx = {**x, **y}
        self.forward(ctx)
        self.loss.decode(ctx)
        return ctx







