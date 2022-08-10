"""Model store which provides pretrained models."""
import os
import logging
import torch
from collections import OrderedDict

from utilities.files import download

__all__ = ["get_model_file", "purge", "ModelLoader"]


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    "vovnet39": "https://dl.dropbox.com/s/q98pypf96rhtd8y/vovnet39_ese_detectron2.pth",
    "dla34": 'http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth'
}


def get_url(name):
    url = model_urls.get(name, None)
    if url is None:
        raise ValueError(
            "Pretrained model for {name} is not available.".format(name=name))
    return url


def get_model_file(name, root=os.path.join("~", ".encoding", "models")):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot
     be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    url = get_url(name)
    file_name = url.split('/')[-1]
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name)
    if os.path.exists(file_path):
        return file_path
    else:
        print("Model file is not found. Downloading.")

    if not os.path.exists(root):
        os.makedirs(root)

    download(url, path=file_path, overwrite=True)

    return file_path


def purge(root=os.path.join("~", ".encoding", "models")):
    r"""Purge all pretrained model files in local file store.

    Parameters
    ----------
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".pth"):
            os.remove(os.path.join(root, f))


class ModelLoader:

    def __call__(self, model, cfg, tasks, log_msg=True, model_name='uninet'):
        self.model = model
        self.log_msg = log_msg
        self.cfg = cfg
        pretrained_path = cfg.MODEL.PRETRAINED_PATH
        is_full_model = cfg.MODEL.IS_FULL_MODEL
        if is_full_model:
            loaded = torch.load(pretrained_path)['model']
            model.load_state_dict(loaded, strict=False)
            self.log('Loaded full model')
            return model
        if pretrained_path == '':
            logging.info('No models loaded using pretrained path')
            return model

        if model_name != 'cross_stitch':
            self.backbone = self.model.base_net.backbone
        self.neck = None
        if hasattr(self.model, 'encoder_decoder') and hasattr(
                self.model.encoder_decoder, 'neck'):
            self.neck = self.model.encoder_decoder.neck
        elif hasattr(self.model, 'neck'):
            self.neck = self.model.neck

        self.load_backbone = cfg.MODEL.LOAD_BACKBONE
        self.backbone_load_name = cfg.MODEL.BACKBONE_LOAD_NAME
        backbone_to = [("backbone.bottom_up.", "")]
        neck_to, head_to = [], []

        self.loaded = torch.load(pretrained_path)
        if 'model' in self.loaded.keys():
            self.loaded = self.loaded['model']
        if 'state_dict' in self.loaded.keys():
            self.loaded = self.loaded['state_dict']
        self.det_head = None
        if 'detect' in tasks:
            if model_name != 'uninet':
                self.det_head = self.model.heads['detect']
            else:
                self.det_head = self.model.head.det_head
            self.neck_names = cfg.MODEL.NECK_LOAD_NAMES
            self.head_name = cfg.MODEL.HEAD_LOAD_NAME
            neck_to = [("backbone.", ""), ("fpn_lateral3", "lateral_convs.0"),
                       ("fpn_lateral4", "lateral_convs.1"),
                       ("fpn_lateral5", "lateral_convs.2"),
                       ("fpn_output3", "fpn_convs.0"),
                       ("fpn_output4", "fpn_convs.1"),
                       ("fpn_output5", "fpn_convs.2")]
            head_to = [("proposal_generator.fcos_head", "head"),
                       ("ctrness", "centerness")]

        if model_name == 'cross_stitch':
            self.backbone = self.model.bb_segment.backbone
            self.load_weights(backbone_to, neck_to, head_to)
            self.backbone = self.model.bb_depth.backbone
            self.load_weights(backbone_to, neck_to, head_to)
            if 'detect' in tasks:
                self.backbone = self.model.bb_detect.backbone
                self.load_weights(backbone_to, neck_to, head_to)
        else:
            self.load_weights(backbone_to, neck_to, head_to)

        return model

    def replace_weight(self, mod_name, weight, part, to):
        replaced_name = mod_name
        for replace_to in to:
            replaced_name = replaced_name.replace(*replace_to)
        if replaced_name in part.state_dict():
            if part.state_dict()[replaced_name].size() != weight.size():
                self.log(f'Size not matching: {replaced_name}')
                return None
            part.state_dict()[replaced_name].copy_(weight)
        else:
            self.log(f'{replaced_name} not in model..')

    def load_weights(self, backbone_to, neck_to, head_to):
        for mod_name, weight in self.loaded.items():
            if self.neck is not None and any(
                    [n in mod_name for n in self.neck_names]):
                self.replace_weight(mod_name, weight, self.neck, neck_to)
            elif self.backbone_load_name in mod_name:
                if self.load_backbone:
                    self.replace_weight(mod_name, weight,
                                        self.backbone, backbone_to)
            elif self.det_head is not None and self.head_name in mod_name:
                self.replace_weight(mod_name, weight,
                                    self.det_head, head_to)
            else:
                self.log(rf"unexpected weight: {mod_name}")

    def log(self, msg):
        if self.log_msg:
            logging.info(msg)
