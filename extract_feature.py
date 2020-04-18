import itertools
import os.path as osp
import argparse
import cv2
import h5py
from urllib.request import urlretrieve
import glob
import numpy as np

from model.resunet import ResUNetBN2D2

import torch


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


def dump_feature(config):
  img_glob = osp.join(config.source, '*/*/images/*.jpg')
  imgs = glob.glob(img_glob)
  print(f'grab {len(imgs)} images')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  checkpoint = torch.load(config.weights)
  model = ResUNetBN2D2(1, 64, normalize_feature=True)
  model.load_state_dict(checkpoint['state_dict'])
  model.eval()
  model = model.to(device)

  num_kp = config.num_kp
  desc_name = f'ucn-{num_kp}'

  for img_path in imgs:
    img = prep_image(img_path)
    F = model(to_normalized_torch(img, device))
    filename = f'{img_path}.{desc_name}.h5'
    F = F[0]
    F = F.permute(2, 1, 0)

    nx, ny, nc = F.shape
    kp = np.asarray(list(itertools.product(np.arange(nx), np.arange(ny))))
    desc = F.reshape(-1, nc).cpu().numpy()

    kp = random_sample(kp, num_kp)
    desc = random_sample(desc, num_kp)

    with h5py.File(filename, 'w') as fp:
      fp.create_dataset('kp', kp.shape, dtype=np.float32)
      fp.create_dataset('desc', desc.shape, dtype=np.float32)
      fp['kp'][:] = kp
      fp['desc'][:] = desc
  print(f"save {filename}")


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
  config = parser.parse_args()

  if not osp.isfile('ResUNetBN2D2-YFCC100train.pth'):
    print('Downloading weights...')
    urlretrieve(
        "https://node1.chrischoy.org/data/publications/ucn/ResUNetBN2D2-YFCC100train-100epoch.pth",
        'ResUNetBN2D2-YFCC100train.pth')

  with torch.no_grad():
    dump_feature(config)
