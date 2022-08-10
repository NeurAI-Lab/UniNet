from encoding_custom.models.uninet import get_uninet
from encoding_custom.models.mtan import get_mtan
from encoding_custom.models.cross_stitch import get_cross_stitch
from encoding_custom.models.padnet import get_padnet
from encoding_custom.models.mti_net import get_mtinet
from encoding_custom.models.baselines import get_single_task_baseline, get_mtl_baseline


def get_multitask_model(name, **kwargs):
    models = {
        'uninet': get_uninet,
        'mtan': get_mtan, 'cross_stitch': get_cross_stitch,
        'padnet': get_padnet, 'mtinet': get_mtinet,
        'single': get_single_task_baseline, 'multi': get_mtl_baseline}
    return models[name.lower()](**kwargs)
