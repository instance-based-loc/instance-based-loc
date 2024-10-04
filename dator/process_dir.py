import os 
import glob 
import re 
import os.path as osp 



def _process_dir(dir_path, relabel=False):
    img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
    pattern = re.compile(r'([-\d]+)_c(\d)')

    pid_container = set()
    for img_path in sorted(img_paths):
        print(f"img_path: {img_path}")
        pid, _ = map(int, pattern.search(img_path).groups())
        print(pattern.search(img_path).groups())
        print(pid)
        if pid == -1: continue  # junk images are just ignored
        pid_container.add(pid)
        break
    # pid2label = {pid: label for label, pid in enumerate(pid_container)}
    # dataset = []
    # for img_path in sorted(img_paths):
    #     pid, camid = map(int, pattern.search(img_path).groups())
    #     if pid == -1: continue  # junk images are just ignored
    #     assert 0 <= pid <= 1501  # pid == 0 means background
    #     assert 1 <= camid <= 6
    #     camid -= 1  # index starts from 0
    #     if relabel: pid = pid2label[pid]
    #
    #     # dataset.append((img_path, self.pid_begin + pid, camid, 1))
    # return dataset

_process_dir("data/market1501/bounding_box_train", relabel=True)
