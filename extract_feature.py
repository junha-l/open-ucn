import argparse
import glob
import itertools
import os.path as osp
from urllib.request import urlretrieve

import cv2
import h5py
import numpy as np
import torch

from lib.util_2d import normalize_keypoint
from model.resunet import ResUNetBN2D2
from util.file import loadh5
from util.visualization import visualize_image_correspondence


def prep_image(full_path):
  assert osp.exists(full_path), f"File {full_path} does not exist."
  return cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)


def to_normalized_torch(img, device):
  """
  Normalize the image to [-0.5, 0.5] range and augment batch and channel dimensions.
  """
  img = img.astype(np.float32) / 255 - 0.5
  return torch.from_numpy(img).to(device)[None, None, :, :]


def random_sample(arr, n):
  np.random.seed(0)
  total = len(arr)
  num_sample = min(total, n)
  idx = sorted(np.random.choice(range(total), num_sample, replace=False))
  return np.asarray(arr)[idx]


def dump_correspondences(config):
  # load model
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  checkpoint = torch.load(config.weights)
  model = ResUNetBN2D2(1, 64, normalize_feature=True)
  model.load_state_dict(checkpoint['state_dict'])
  model.eval()
  model = model.to(device)
  print("load model")

  # load dataset
  source = config.source
  with h5py.File(config.yfcc_path, 'r+') as ofp:
    len_dset = len(ofp['coords'])
    print("len dataset : ", len_dset)

    new_data = {}
    new_data['ucn_coords'] = ofp.create_group('ucn_coords')
    new_data['ucn_n_coords'] = ofp.create_group('ucn_n_coords')

    for i in range(len_dset):
      idx = str(i)
      coords = ofp['coords'][idx]
      img_path0 = coords.attrs['img0']
      img_path1 = coords.attrs['img1']
      img_idx0 = int(coords.attrs['idx0']) + 1
      img_idx1 = int(coords.attrs['idx1']) + 1
      print(f"[{i}/{len_dset}] {img_path0}, {img_path1}")

      # load & prepare calibration
      calib_path0 = "/".join(img_path0.split("/")[:-2])
      calib_path0 += f"/calibration/calibration_{img_idx0:06d}.h5"
      calib_path1 = "/".join(img_path1.split("/")[:-2])
      calib_path1 += f"/calibration/calibration_{img_idx1:06d}.h5"

      calib0 = loadh5(osp.join(source, calib_path0))
      calib1 = loadh5(osp.join(source, calib_path1))
      K0, K1 = calib0['K'], calib1["K"]
      imsize0, imsize1 = calib0['imsize'], calib1['imsize']

      # load images & extract features
      img0 = prep_image(osp.join(source, img_path0))
      img1 = prep_image(osp.join(source, img_path1))
      F0 = model(to_normalized_torch(img0, device))
      F1 = model(to_normalized_torch(img1, device))

      # build correspondences
      x0, y0, x1, y1 = visualize_image_correspondence(
          img0, img1, F0[0], F1[0], i, mode='gpu-all', config=config, visualize=False)

      kp0 = np.stack((x0, y0), 1)
      kp1 = np.stack((x1, y1), 1)

      kp0 = random_sample(kp0, config.num_kp)
      kp1 = random_sample(kp1, config.num_kp)
      norm_kp0 = normalize_keypoint(kp0, K0, imsize0 * 0.5)[:, :2]
      norm_kp1 = normalize_keypoint(kp1, K1, imsize1 * 0.5)[:, :2]

      coords = np.concatenate((kp0, kp1), axis=1)
      n_coords = np.concatenate((norm_kp0, norm_kp1), axis=1)

      coords_data = new_data['ucn_coords'].create_dataset(
          str(i), coords.shape, dtype=np.float32)
      coords_data[:] = coords.astype(np.float32)
      n_coords_data = new_data['ucn_n_coords'].create_dataset(
          str(i), n_coords.shape, dtype=np.float32)
      n_coords_data[:] = n_coords.astype(np.float32)
      print(f"[{i}/{len_dset}] save {coords.shape}, {n_coords.shape} coordinate")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--weights',
      default='ResUNetBN2D2-YFCC100train.pth',
      type=str,
      help='Path to pretrained weights')
  parser.add_argument(
      '--source',
      type=str,
      required=True,
      help="source directory of YFCC100M dataset",
  )
  parser.add_argument(
      '--target',
      type=str,
      required=True,
      help="target directory to save processed data",
  )
  parser.add_argument(
      '--num_kp',
      type=int,
      default=10000,
  )
  parser.add_argument(
      '--yfcc_path',
      type=str,
  )
  parser.add_argument(
      '--ucn_inlier_threshold_pixel',
      type=float,
      default=4,
      help="Inlier threshold for hit test")
  parser.add_argument(
      '--nn_max_n',
      type=int,
      default=50
      )
  config = parser.parse_args()

  if not osp.isfile('ResUNetBN2D2-YFCC100train.pth'):
    print('Downloading weights...')
    urlretrieve(
        "https://node1.chrischoy.org/data/publications/ucn/ResUNetBN2D2-YFCC100train-100epoch.pth",
        'ResUNetBN2D2-YFCC100train.pth')

  with torch.no_grad():
    dump_correspondences(config)
