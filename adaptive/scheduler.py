import math

def beta_warmup(epoch, max_epoch, beta_max):

    return beta_max * (1 - math.exp(-3 * epoch / max_epoch))