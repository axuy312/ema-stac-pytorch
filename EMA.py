import copy, torch

class EMA():
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay

    def update(self, model):
        for name, param in self.model.named_parameters():
            assert name in model.state_dict()
            param.data = (1.0 - self.decay) * model.state_dict()[name] + self.decay * param.data
