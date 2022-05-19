import numpy as np
import torch
import cv2 as cv
import glob
import os
import argparse
from icecream import ic

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import load_model
from utils.config import load_cfg
from utils.utils import get_mano_path, imgUtils
from dataset.dataset_utils import IMG_SIZE
from core.test_utils import InterRender


def cut_img(img, bbox):
    cut = img[max(int(bbox[2]), 0):min(int(bbox[3]), img.shape[0]),
              max(int(bbox[0]), 0):min(int(bbox[1]), img.shape[1])]
    cut = cv.copyMakeBorder(cut,
                            max(int(-bbox[2]), 0),
                            max(int(bbox[3] - img.shape[0]), 0),
                            max(int(-bbox[0]), 0),
                            max(int(bbox[1] - img.shape[1]), 0),
                            borderType=cv.BORDER_CONSTANT,
                            value=(0, 0, 0))
    return cut


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='misc/model/config.yaml')
    parser.add_argument("--model", type=str, default='misc/model/wild_demo.pth')
    parser.add_argument("--live_demo", action='store_true')
    parser.add_argument("--img_path", type=str, default='demo/')
    parser.add_argument("--save_path", type=str, default='demo/')
    parser.add_argument("--render_size", type=int, default=256)
    opt = parser.parse_args()

    model = InterRender(cfg_path=opt.cfg,
                        model_path=opt.model,
                        render_size=opt.render_size)

    video_reader = cv.VideoCapture(opt.img_path)
    input_vid_base = os.path.splitext(os.path.split(opt.img_path)[1])[0]

    output_vid_name = os.path.join(opt.save_path, input_vid_base + '_output.avi')

    width = int(video_reader.get(cv.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(video_reader.get(cv.CAP_PROP_FRAME_HEIGHT))  # float `height`
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps = video_reader.get(cv.CAP_PROP_FPS)
    # fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # video_reader.set(cv.CAP_PROP_FOURCC, fourcc)
    out = cv.VideoWriter(output_vid_name, fourcc, fps, (width,  height))

    smooth = False
    params_last = None
    params_last_v = None
    params_v = None
    params_a = None

    fIdx = 0
    with torch.no_grad():
        while True:
            fIdx = fIdx + 1
            _, img = video_reader.read()
            if img is None:
                exit()
            w = min(img.shape[1], img.shape[0]) / 2 * 1.0
            left = int(img.shape[1] / 2 - w)
            top = int(img.shape[0] / 2 - w)
            size = int(2 * w)
            bbox = [left, left + size, top, top + size]
            bbox = np.array(bbox).astype(np.int32)
            crop_img = img[bbox[2]:bbox[3], bbox[0]:bbox[1]]
            # ic(crop_img.shape)
            params = model.run_model(crop_img)
            if smooth and params_last is not None and params_v is not None and params_a is not None:
                for k in params.keys():
                    if isinstance(params[k], torch.Tensor):
                        pred = params_last[k] + params_v[k] + 0.5 * params_a[k]
                        params[k] = (0.7 * params[k] + 0.3 * pred)

            img_out = model.render(params, bg_img=crop_img)
            img[bbox[2]:bbox[3], bbox[0]:bbox[1]] = cv.resize(img_out, (size, size))
            cv.line(img, (int(bbox[0]), int(bbox[2])), (int(bbox[0]), int(bbox[3])), (0, 0, 255), 2)
            cv.line(img, (int(bbox[1]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), (0, 0, 255), 2)
            cv.line(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[2])), (0, 0, 255), 2)
            cv.line(img, (int(bbox[0]), int(bbox[3])), (int(bbox[1]), int(bbox[3])), (0, 0, 255), 2)
            # cv.imshow('cap', img)
            out.write(img)

            if params_last is not None:
                params_v = {}
                for k in params.keys():
                    if isinstance(params[k], torch.Tensor):
                        params_v[k] = (params[k] - params_last[k])
            if params_last_v is not None and params_v is not None:
                params_a = {}
                for k in params.keys():
                    if isinstance(params[k], torch.Tensor):
                        params_a[k] = (params_v[k] - params_last_v[k])
            params_last = params
            params_last_v = params_v

    out.release()