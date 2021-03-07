class AverageMeter(object):
    """ Computes and stores the average and current value"""
    def __init__(self, name: str, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.cout = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cout += n
        self.avg = self.sum / self.cout

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    
    meter = AverageMeter("default")
    for i in range(5):
        meter.update(i * 2)
    import pdb; pdb.set_trace()
    print(meter)

