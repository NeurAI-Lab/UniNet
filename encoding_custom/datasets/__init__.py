import logging

from encoding_custom.datasets.uninet_cs import CityscapesLoader
from encoding_custom.datasets.nyuv2 import NYUv2Loader


datasets = {
    'uninet_cs': CityscapesLoader,
    'nyuv2': NYUv2Loader,
}

acronyms = {
    'uninet_cs': 'Cityscapes dataset',
    'nyuv2': 'nyuv2 dataset',
}


def get_multitask_dataset(name, **kwargs):
    logging.info(f'Dataset loaded is {name}')
    return datasets[name.lower()](**kwargs)


def get_num_classes(name):
    return datasets[name.lower()].NUM_CLASSES


def get_instance_names(name):
    if hasattr(datasets[name.lower()], 'INSTANCE_NAMES'):
        return datasets[name.lower()].INSTANCE_NAMES
    else:
        return []


def get_semantic_names(name):
    if hasattr(datasets[name.lower()], 'SEMANTIC_NAMES'):
        return datasets[name.lower()].SEMANTIC_NAMES
    else:
        # return instance names if semantic names are not available.
        return get_instance_names(name)
