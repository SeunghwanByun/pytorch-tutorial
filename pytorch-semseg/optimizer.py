import logging

from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

logger = logging.getLogger("Segmentation")

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}

def get_optimizer(opt):
    if opt is None:
        logger.info("Using SGD optimizer")
        return SGD
    else:
        if opt not in key2opt:
            raise NotImplementedError("Optimizer {} not implemented.".format(opt))

        logger.info("Using {} optimizer".format(opt))
        return key2opt[opt]