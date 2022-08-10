import numpy as np
import matplotlib.pyplot as plt
import torch


class ReadMEInstParams:

    def __init__(self, cfg):
        self.cfg = cfg
        self.encoding_type = cfg.MODEL.MEINST.ENCODING_TYPE
        assert self.encoding_type in ['implicit', 'explicit'], \
            'unknown encoding type'
        self.sigmoid = cfg.MODEL.MEINST.SIGMOID
        self.whiten = cfg.MODEL.MEINST.WHITEN
        self.num_components = cfg.MODEL.MEINST.NUM_COMPONENTS
        self.encoding_dim = (cfg.MODEL.MEINST.ENCODING_DIM,
                             cfg.MODEL.MEINST.ENCODING_DIM)
        self.input_scale = cfg.INPUT.IMAGE_SIZE

        self.pca_params = None
        self.t = None
        self.w = None

    def load_pca(self, is_ts=False, device=None):
        pca_path = self.cfg.MODEL.MEINST.PCA_PATH
        pca_params = np.load(pca_path)
        self.pca_params = {attr: pca_params[attr]
                           for attr in pca_params.files}
        for attr in ['components_', 'explained_variance_',
                     'explained_variance_ratio_']:
            self.pca_params[attr] = self.pca_params[
                                        attr][:self.num_components]
        self.t = self.pca_params['components_']
        self.w = self.t.T
        if is_ts:
            device = device if device is not None else self.cfg.DEVICE
            self.pca_params = {
                key: torch.from_numpy(value).float().to(device)
                for key, value in self.pca_params.items()}
            self.t = torch.from_numpy(self.t).to(device)
            self.w = torch.from_numpy(self.w).to(device)


def apply_inverse_sigmoid(data):
    VALUE_MAX = 0.05
    VALUE_MIN = 0.01
    value_random = VALUE_MAX * np.random.rand(data.shape[0], data.shape[1])
    value_random = np.maximum(value_random, VALUE_MIN)
    data = np.where(data > value_random, 1 - value_random, value_random)
    data = -1 * np.log((1 - data) / data)

    return data


def apply_sigmoid(data):
    return 1./(1.+1./np.exp(data))


def mask_encode(flat_mask, pca_params, sigmoid=False, whiten=False):
    if sigmoid:
        flat_mask = apply_inverse_sigmoid(flat_mask[None, :])
        flat_mask = np.squeeze(flat_mask)

    if 'mean_' in pca_params.keys():
        flat_mask = flat_mask - pca_params['mean_']
    flat_mask = np.matmul(pca_params['components_'], flat_mask)
    if whiten:
        flat_mask /= np.sqrt(pca_params['explained_variance_'])

    return flat_mask


def mask_decode(mask_batch, pca_params, sigmoid=False, whiten=False):
    components_ = pca_params['components_'].clone()
    if whiten:
        components_ *= torch.sqrt(
            pca_params['explained_variance_'].unsqueeze(1))
    mask_batch = torch.matmul(mask_batch, components_)
    if 'mean_' in pca_params.keys():
        mask_batch = mask_batch + pca_params['mean_']

    if sigmoid:
        mask_batch = torch.sigmoid(mask_batch)
    else:
        mask_batch = torch.clamp(mask_batch, min=0.01, max=0.99)

    return mask_batch


def plot(x, y, xlabel, ylabel, title=None, xticks=None, yticks=None,
         marker=None, y_txt=False, xlim=None, ylim=None, save_path=None):
    plt.plot(x, y, marker=marker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(0, xlim)
    if ylim is not None:
        plt.ylim(0, ylim)
    plt.xticks(xticks)
    plt.yticks(yticks)
    if y_txt:
        for x1, y1 in zip(x, y):
            plt.text(x1 - 1, y1 + 0.2, str(round(y1, 1)), fontsize=10)
    if title is not None:
        plt.title(title)
    plt.grid(axis='y')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def bbox_to_boxcoords(bbox, im_shape):
    x1, y1 = bbox[:2]
    x2, y2 = x1 + bbox[2], y1 + bbox[3]
    x1, x2 = min(x1, im_shape[1] - 1), min(x2, im_shape[1])
    y1, y2 = min(y1, im_shape[0] - 1), min(y2, im_shape[0])

    x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)

    return [x1, y1, x2, y2]
