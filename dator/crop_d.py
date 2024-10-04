import os
import shutil
import os.path as osp
import sys
import cv2
from tqdm import tqdm


if not os.path.exists(f"datav1"):
    os.mkdir(f"datav1")
else:
    print(f"datav1 already exists!")
    sys.exit(1)


for coarse in tqdm(os.listdir("lora")):
    for fine in os.listdir(osp.join("lora", coarse)):
        classname = f"{coarse}{fine}"
        os.mkdir(f"datav1/{classname}")
        # count of images in the above directory
        count = 0
        files = os.listdir(osp.join("lora", coarse, fine, "croppedrgb"))
        image_files = sorted([file for file in files if file.find(".png") != -1])
        text_files = sorted([file for file in files if file.find(".txt") != -1])
        for image, text in zip(image_files, text_files):
            shutil.copy(
                osp.join("lora", coarse, fine, "croppedrgb", image),
                f"datav1/{classname}/{count}.jpg"
            )
            depth_path = osp.join("lora", coarse, fine, "depth")
            with open(osp.join("lora", coarse, fine, "croppedrgb", text), "r") as f:
                tuple_string = f.read().strip()
            tuple_string = tuple_string.strip("()")
            tuple_elements = tuple_string.split(",")

            tuple_elements = [
                int(element) if element.strip().isdigit() else element.strip()
                for element in tuple_elements
            ]
            assert len(tuple_elements) == 4 
            x1, y1, x2, y2 = tuple_elements
            # print(tuple_elements)
            img = cv2.imread(osp.join(depth_path, image))
            img = img[y1:y2, x1:x2]
            cv2.imwrite(f"datav1/{classname}/{count}_d.jpg", img)
            count += 1
