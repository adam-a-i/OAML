from __future__ import print_function, absolute_import
import os.path as osp
import os
from glob import glob


class Face(object):

    def __init__(self, root):
        print(f'Loading dataset from {root}')
        self.images_dir = osp.join(root)
        self.data = []
        self.num_ids = 0
        self.load()

    def preprocess(self):
        # Get all class directories
        class_dirs = [d for d in os.listdir(self.images_dir) 
                     if os.path.isdir(os.path.join(self.images_dir, d))]
        print(f'Found {len(class_dirs)} class directories')
        
        data = []
        all_pids = {}

        for class_dir in sorted(class_dirs):
            # Get all images in this class directory
            class_path = os.path.join(self.images_dir, class_dir)
            image_files = glob(os.path.join(class_path, '*.jpg'))
            
            for fpath in sorted(image_files):
                fname = osp.basename(fpath)
                # Extract person ID from filename (e.g. S2-P1-F-39-1.jpg -> S2-P1)
                pid = '-'.join(fname.split('-')[:2])  # Get the first two parts

                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
                data.append((os.path.join(class_dir, fname), all_pids[pid]))

        return data, len(all_pids)

    def load(self):
        self.data, self.num_ids = self.preprocess()

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  data    | {:5d} | {:8d}"
              .format(self.num_ids, len(self.data)))

    def get_image_path(self, fname):
        return osp.join(self.images_dir, fname)