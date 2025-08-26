from omegaconf import ListConfig

def SumResolver(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple, ListConfig)):
        return int(sum(args[0]))
    return int(sum(args))