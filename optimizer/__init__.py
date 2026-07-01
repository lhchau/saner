from torch.optim import SGD
from .sam import SAM
from .samonly import SAMONLY
from .samwo import SAMWO
from .saner import SANER
from .fsam import FriendlySAM
from .fsaner import FriendlySANER
from .vasso import VASSO
from .vassosaner import VASSOSANER
from .gsam import GSAM
from .gsaner import GSANER
from .saner_last import SANERLAST

def get_optimizer(
    net,
    opt_name='sam',
    opt_hyperpara={}):
    if opt_name == 'sam':
        return SAM(net.parameters(), **opt_hyperpara)
    elif opt_name == 'sgd':
        return SGD(net.parameters(), **opt_hyperpara)
    elif opt_name == 'samonly':
        return SAMONLY(net.parameters(), **opt_hyperpara)
    elif opt_name == 'samwo':
        return SAMWO(net.parameters(), **opt_hyperpara)
    elif opt_name == 'saner':
        return SANER(net.parameters(), **opt_hyperpara)
    elif opt_name == 'fsam':
        return FriendlySAM(net.parameters(), **opt_hyperpara)
    elif opt_name == 'fsaner':
        return FriendlySANER(net.parameters(), **opt_hyperpara)
    elif opt_name == 'vasso':
        return VASSO(net.parameters(), **opt_hyperpara)
    elif opt_name == 'vassosaner':
        return VASSOSANER(net.parameters(), **opt_hyperpara)
    elif opt_name == 'gsam':
        return GSAM(net.parameters(), **opt_hyperpara)
    elif opt_name == 'gsaner':
        return GSANER(net.parameters(), **opt_hyperpara)
    elif opt_name == 'sanerlast':
        return SANERLAST(net.parameters(), **opt_hyperpara)
    else:
        raise ValueError("Invalid optimizer!!!")