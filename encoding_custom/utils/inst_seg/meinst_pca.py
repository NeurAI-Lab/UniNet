import os
import cv2
import numpy as np
import logging
import pickle
import argparse
from tqdm import tqdm
import torch
from pycocotools.coco import COCO

from configs.defaults import _C as cfg
import encoding_custom.utils.inst_seg.meinst_utils as utils
from encoding_custom.evaluation.evaluators import SegmentEvaluator
from utilities.train_utils import mkdir


def has_valid_annotation(anno):
    if len(anno) == 0:
        return False
    if all(not (obj.get("iscrowd", 0) == 0 and
                obj.get("real_bbox", True)) for obj in anno):
        return False
    return True


class EncodeInstanceMasks:
    def __init__(self, encode_dim=(28, 28)):
        (self.data_path, self.pca_path,
         self.components_path) = [os.path.join(args.results_path, f) if f is
                                  not None else None for f in (args.data_file,
                                  args.pca_file, args.components_file)]
        self.coco = COCO(args.json_path)
        self.image_ids = self.coco.getImgIds()
        ids = []
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)

        self.image_ids = ids

        self.encode_dim = encode_dim
        self.pca = None
        self.sigmoid = args.sigmoid
        self.whiten = args.whiten

    def get_anns(self, im_id):
        anns_id = self.coco.getAnnIds(imgIds=im_id)
        return self.coco.loadAnns(anns_id)

    def arrange_data(self):
        if os.path.isfile(self.data_path):
            return np.load(self.data_path)

        data = []
        for im_id in tqdm(self.image_ids, desc='Processing instance masks'):
            anns = self.get_anns(im_id)
            for ann in anns:
                # remove boxes with no width or height (float)
                if ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
                    continue
                mask = self.coco.annToMask(ann)
                bcoords = utils.bbox_to_boxcoords(ann['bbox'], mask.shape)
                x0, y0, x1, y1 = [int(bc) for bc in bcoords]
                mask = mask[y0:y1, x0:x1]
                mask = cv2.resize(mask, self.encode_dim,
                                  interpolation=cv2.INTER_NEAREST)
                mask = np.array(mask, dtype=np.float32)
                if np.all(mask == 0):
                    continue
                data.append(mask.flatten())

        data = np.stack(data, axis=0)
        if self.data_path is not None:
            np.savez(self.data_path, inst_masks=data.astype(np.float16))

        return {'inst_masks': data}

    def encode(self):
        from sklearn.decomposition import PCA
        data = self.arrange_data()['inst_masks']
        if self.sigmoid:
            data = utils.apply_inverse_sigmoid(data)
        self.pca = PCA(n_components=None, whiten=self.whiten)
        self.pca.fit(data)

        if self.pca_path is not None:
            with open(self.pca_path, 'wb') as f:
                pickle.dump(self.pca, f)

    def save_components(self):
        to_save = ['components_', 'explained_variance_',
                   'explained_variance_ratio_', 'mean_']
        to_save = {attr: getattr(self.pca, attr) for attr in to_save}
        np.savez(self.components_path, **to_save)

    def calc_recon_error(self, num_components, save_components=True,
                         plot=False):
        data = self.arrange_data()['inst_masks']
        masks = data.copy()

        if self.pca is None:
            if os.path.isfile(self.pca_path):
                with open(self.pca_path, 'rb') as f:
                    self.pca = pickle.load(f)
            else:
                self.encode()
        if save_components:
            self.save_components()

        if self.sigmoid:
            data = utils.apply_inverse_sigmoid(data)
        masks = np.where(masks >= 0.5, 1, 0)

        components = self.pca.components_.copy()
        explained_variance = self.pca.explained_variance_
        recon_error = []
        for num in num_components:
            self.pca.components_ = components[:num, :]
            self.pca.explained_variance_ = explained_variance[:num]
            encoding = self.pca.transform(data)
            reconstruct_masks = self.pca.inverse_transform(encoding)

            if self.sigmoid:
                reconstruct_masks = utils.apply_sigmoid(reconstruct_masks)
            reconstruct_masks = np.where(reconstruct_masks >= 0.5, 1, 0)

            evaluator = SegmentEvaluator(cfg, num_classes=2)
            indices = np.arange(16, len(reconstruct_masks), 16)
            reconstruct_masks = np.split(
                reconstruct_masks, indices_or_sections=indices)
            targets = np.split(masks, indices_or_sections=indices)
            for tar, pred in tqdm(zip(targets, reconstruct_masks),
                                  desc='Calculating mIoU'):
                tar = torch.from_numpy(tar)
                # one hot encode prediction..
                pred = torch.from_numpy(pred)[:, None, :]
                pred = torch.cat((1 - pred, pred), dim=1)
                evaluator.process(tar, pred, None)

            mean_iou = evaluator.evaluate()['key_metrics/segment_MIoU']
            recon_error.append(100 - mean_iou)

        save_path = None
        if not plot:
            save_path = os.path.join(args.results_path, 'recon_error.png')
        utils.plot(num_components, recon_error, 'Number of components',
                   'Reconstruction error (1 - mIoU) %',
                   xticks=num_components, title='MEInst recon error',
                   marker='.', y_txt=True, ylim=20, save_path=save_path)

        self.pca.components_ = components
        return recon_error


def main():
    results_path = args.results_path
    mkdir(results_path)
    num_components = np.arange(10, 101, 10)
    encode_masks = EncodeInstanceMasks()
    encode_masks.calc_recon_error(num_components)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    cfg.MISC.LOG_PER_CLASS_METRICS = False
    cfg.INSTANCE_NAMES = []
    cfg.SEMANTIC_NAMES = []
    cfg.freeze()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'json_path', help='path which contains the coco format json file')
    parser.add_argument('--results_path', help='path to save the results',
                        default='./')
    parser.add_argument('--data_file', help='data file name',
                        default='inst_masks.npz')
    parser.add_argument('--pca_file', help='pca pickle file name',
                        default='pca_learned.pkl')
    parser.add_argument('--components_file', help='components file name',
                        default='pca_params.npz')
    parser.add_argument('--sigmoid', action='store_true',
                        help='use sigmoid transform', default=True)
    parser.add_argument('--whiten', action='store_true',
                        help='use whiten', default=True)
    args = parser.parse_args()
    main()
