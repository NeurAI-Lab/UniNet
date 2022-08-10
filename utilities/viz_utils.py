import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from encoding_custom.utils.inst_seg.inference import get_inst_mask


# colour map
label_colours_global = []
label_colours_global_det = []

city_seg_colors = np.asarray([[128, 64, 128], [232, 35, 244], [70, 70, 70],
                              [156, 102, 102], [153, 153, 190], [153, 153, 153],
                              [30, 170, 250], [0, 220, 220], [35, 142, 107],
                              [152, 251, 152], [180, 130, 70], [60, 20, 220],
                              [0, 0, 255], [142, 0, 0], [70, 0, 0], [100, 60, 0],
                              [100, 80, 0], [230,  0, 0], [32, 11, 119],
                              [255, 255, 255]], dtype=np.uint8)


def hex_to_rgb(hex_val):
    hex_val = hex_val.lstrip('#')
    hlen = len(hex_val)
    return tuple(int(hex_val[l:l + hlen // 3], 16)
                 for l in range(0, hlen, hlen // 3))


def rgb_to_rgb(hex_val):
    hlen = len(hex_val)
    return tuple(int(hex_val[l:l + hlen // 3], 16)
                 for l in range(0, hlen, hlen // 3))


colors = np.load('utilities/extra/colors.npy')
for i in colors:
    label_colours_global.append(hex_to_rgb(str(i)))
detcolors = np.load('utilities/extra/palette.npy')
for i in range(0, len(detcolors), 3):
    det = tuple((int(detcolors[i]), int(detcolors[i+1]), int(detcolors[i+2])))
    label_colours_global_det.append(det)


def draw_instances(instances, image, cfg, score_threshold=0.5, alpha=0.4,
                   txt_size=0.35):
    if len(instances) == 0:
        return image
    pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
    pred_classes = instances.pred_classes.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    pred_masks = getattr(instances, 'pred_masks', None)
    inst_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    image_h, image_w = image.shape[:2]
    pred_h, pred_w = instances.image_size
    width, height = image_w / pred_w, image_h / pred_h

    colors_arr = np.array(label_colours_global, dtype=np.uint8)
    unique_labels = {cls: [lbl for lbl in range(1, 110)]
                     for cls in range(cfg.NUM_CLASSES.DETECT)}
    for idx, (box, label, score) in enumerate(zip(
            pred_boxes, pred_classes, scores)):
        if score >= score_threshold:
            un_lbl = unique_labels[label].pop(0)
            box = [box[0] * width, box[1] * height, (box[2] - box[0]) * width,
                   (box[3] - box[1]) * height]
            x0, y0 = int(box[0]), int(box[1])
            x1, y1 = int(box[0] + box[2]), int(box[1] + box[3])
            inst_color = tuple((int(val) for val in colors_arr[un_lbl]))
            cv2.rectangle(image, (x0, y0), (x1, y1), inst_color, thickness=2)
            class_name = cfg.INSTANCE_NAMES[label]
            ((text_width, text_height), _) = cv2.getTextSize(
                class_name, cv2.FONT_HERSHEY_SIMPLEX, txt_size, 1)
            cv2.rectangle(image, (x0, y0 - int(1.3 * text_height)),
                          (x0 + text_width, y0), inst_color, -1)
            cv2.putText(
                image, text=class_name, org=(x0, y0 - int(0.3 * text_height)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=txt_size,
                color=(255, 255, 255), lineType=cv2.LINE_AA)

            if pred_masks is not None:
                m = get_inst_mask(box, pred_masks[idx], image_h, image_w, cfg)
                inst_mask[:, :] += (inst_mask == 0) * (m * un_lbl)

    indices = np.where(inst_mask != 0)
    inst_mask = colors_arr[inst_mask]
    image[indices] = 0.5 * image[indices] + 0.5 * inst_mask[indices]
    image = np.asarray(image, dtype=np.uint8)
    # cv2.addWeighted(inst_mask, alpha, image, 1 - alpha, 0, image)

    return image


def citys_seg_legend():

    names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
             'traffic_light', 'traffic_sign', 'vegetation',
             'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
             'train', 'motorcycle', 'bicycle']
    [plt.plot(0, 0, '-', c=np.flip(city_seg_colors[i], 0) / 255., label=n)
     for i, n in enumerate(names)]

    leg = plt.legend(loc=4, ncol=6, mode="expand",
                     borderaxespad=0., prop={'size': 20})

    for handle, line, text in zip(leg.legendHandles,
                                  leg.get_lines(), leg.get_texts()):
        handle.set_linewidth(15)
        text.set_color(line.get_color())
    plt.xticks([])
    plt.yticks([])
    plt.show()


def to_cv2_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    image = np.asarray(image, dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def seg_color_map(segment, seg_map=False, use_city_colors=True):
    if not seg_map:
        segment = segment.argmax(0).cpu().numpy()
    segment = np.array(segment, dtype=np.uint8)
    if use_city_colors:
        return city_seg_colors[segment]
    else:
        return colors[segment]


def depth_color_map(depth, cfg, depth_scale=None):
    if depth_scale is None:
        depth_scale = cfg.DATALOADER.MAX_DEPTH
    depth = depth.cpu().numpy()
    depth = np.squeeze(depth)
    # visualize disparity...
    depth = 1 - depth
    depth = to_depth_color_map(depth, depth_scale=depth_scale)
    return cv2.cvtColor(depth, cv2.COLOR_RGB2BGR)


def to_depth_color_map(depth, depth_scale=1.):
    depth = (depth * depth_scale).astype(np.uint8)

    vmax = np.percentile(depth, 95)
    normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    depth = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)

    return depth
