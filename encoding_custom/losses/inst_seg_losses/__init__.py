from .meinst_loss import MEInstLossComputation


def build_inst_seg_loss(cfg):
    return MEInstLossComputation(cfg)
