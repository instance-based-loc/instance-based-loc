# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import os

import os.path as osp

from .bases import BaseImageDataset


class RealSense(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """

    # dataset_dir = "/ssd_scratch/cvit/vaibhav/realsense_all_18_light_mixed"
    # dataset_dir = "/ssd_scratch/cvit/vaibhav/tum"
    dataset_dir = "/ssd_scratch/cvit/vaibhav/rrc"

    def __init__(self, root="", verbose=True, pid_begin=0, **kwargs):
        super().__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.query_dir = osp.join(self.dataset_dir, "val")
        self.gallery_dir = osp.join(self.dataset_dir, "test")

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> RealSense (RRC, Light) loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        (
            self.num_train_pids,
            self.num_train_imgs,
            self.num_train_cams,
            self.num_train_vids,
        ) = self.get_imagedata_info(self.train)
        (
            self.num_query_pids,
            self.num_query_imgs,
            self.num_query_cams,
            self.num_query_vids,
        ) = self.get_imagedata_info(self.query)
        (
            self.num_gallery_pids,
            self.num_gallery_imgs,
            self.num_gallery_cams,
            self.num_gallery_vids,
        ) = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        ctg2pid = {}
        count = 0
        for ctg in sorted(os.listdir(dir_path)):
            ctg2pid[ctg] = count 
            count += 1 
        dataset = []
        for ctg in os.listdir(dir_path):
            all_filenames = sorted(os.listdir(osp.join(dir_path, ctg)))
            img_names = [filename for filename in all_filenames if filename.find("rgb") != -1]
            depth_names = [filename for filename in all_filenames if filename.find("depth") != -1]
            img_paths = [osp.join(dir_path, ctg, name) for name in img_names]
            depth_paths = [osp.join(dir_path, ctg, name) for name in depth_names]
            for img_path, depth_path in zip(img_paths, depth_paths): 
                dataset.append((img_path, int(self.pid_begin + int(ctg2pid[ctg])), 0, 1))

        return dataset
