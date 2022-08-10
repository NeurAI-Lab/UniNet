from .target_transform import *
from . import transforms as transforms_file
from .transforms import Compose


def build_transforms(cfg, crop_size, is_train=True, **kwargs):
    if is_train:
        transforms_list = cfg.DATALOADER.TRAIN_TRANSFORMS
    else:
        transforms_list = cfg.DATALOADER.VAL_TRANSFORMS

    all_transforms = []
    for t in transforms_list:
        t_fn = getattr(transforms_file, t, None)
        assert t_fn is not None, f'Transform {t} not found..'
        kwargs.update({'is_train': is_train})
        all_transforms.append(t_fn(cfg, crop_size, **kwargs))

    return Compose(all_transforms)


def build_inst_target_transform(cfg, image_size, **kwargs):
    return FCOSTargetTransform(image_size)


def build_inst_seg_target_transform(cfg, image_size, **kwargs):
    if cfg.DATALOADER.INST_SEG_ENCODING == 'MEINST':
        return MEInstTargetTransform(cfg, image_size, **kwargs)
    else:
        return InstSegTargetTransform(cfg, image_size, **kwargs)
