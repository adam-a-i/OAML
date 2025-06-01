from __future__ import print_function, absolute_import
import os.path as osp
from glob import glob


class Face(object):

    def __init__(self, root):
        print(f'Loading dataset from {root}')
        self.images_dir = osp.join(root)
        self.data = []
        self.num_ids = 0
        self.load()

    def preprocess(self):
        fpaths = sorted(glob(osp.join(self.images_dir, '*.jpg')))
        print(f'Found {len(fpaths)} images in {self.images_dir}')
        data = []
        all_pids = {}

        for fpath in fpaths:
            fname = osp.basename(fpath)
            # Extract person ID from filename (assuming format: personID_*.jpg)
            pid = int(fname.split('_')[0])

            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            data.append((fname, all_pids[pid]))
        return data, len(all_pids)

    def load(self):
        self.data, self.num_ids = self.preprocess()

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  data    | {:5d} | {:8d}"
              .format(self.num_ids, len(self.data)))