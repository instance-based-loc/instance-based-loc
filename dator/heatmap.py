import re
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import PIL

# Load all test images
test_path ="procthor_test"

# categories = os.listdir(test_path)
# test_classes = []
# for ctg in categories:
#     for i in os.listdir(os.path.join(test_path, ctg)):
#         test_classes.append(ctg + "/" + i)
#
# print(test_classes)
# test_classes = [
#     i
#     for i in sorted(test_classes)
#     if (int(re.sub(r"[a-z_/]*", "", i)) in [x for x in range(1, 16)])
# ]
# test_classes = []
for coarse in os.listdir(test_path):
    for fine in os.listdir(osp.join(test_path, coarse)):
        test_classes.append(osp.join(coarse, fine))
print(test_classes)

test_images = [os.listdir(os.path.join(test_path, c)) for c in test_classes]

# print(test_images)
for i in range(len(test_classes)):
    for j in range(len(test_images[i])):
        test_images[i][j] = os.path.join(test_path, test_classes[i], test_images[i][j])

# %%
for i in range(len(test_classes)):
    for j in range(len(test_images[i])):
        test_images[i][j] = PIL.Image.open(test_images[i][j])

# %%
import matplotlib.pyplot as plt

to_show = 10
fig, axes = plt.subplots(
    to_show, len(test_classes), figsize=(to_show * 2.5, len(test_classes) * 2.5)
)

for i in tqdm(range(len(test_classes))):
    for j, img in enumerate(test_images[i][:to_show]):
        ax = axes[j][i]
        ax.imshow(img)
        ax.axis("off")  # Hide axis

plt.show()


# %%
# get all embeddings

w = []

with torch.no_grad():
    for row in test_images:
        r = []
        for i in row:
            im = test_transforms(i.convert("RGB"))
            k = lora_model(
                im.unsqueeze(0).cuda(), output_hidden_states=True
            ).last_hidden_state[0, 0, :]

            r.append(k)
        w.append(torch.stack(r))
w = torch.stack(w).reshape(-1, 768)

# %%

# plt.plot(np.arange(0,1,0.01), np.power(np.arange(0,1,0.01),exponent))

plt.figure(figsize=(15, 15))
plt.imshow(scores, cmap="hot")
plt.colorbar()

num_instances = 6

x_axis_titles = [
    f"{test_classes[i//num_instances]}"
    for i in range(num_instances // 2, num_instances // 2 + len(scores), num_instances)
]
y_axis_titles = [
    f"{test_classes[i//num_instances]}"
    for i in range(num_instances // 2, num_instances // 2 + len(scores), num_instances)
]

plt.xticks(
    range(num_instances // 2, num_instances // 2 + len(scores), num_instances),
    x_axis_titles,
    fontsize=6,
    rotation=45,
    ha="right",
)
plt.yticks(
    range(num_instances // 2, num_instances // 2 + len(scores), num_instances),
    y_axis_titles,
    fontsize=6,
    va="center",
)

for i in range(1, len(scores)):
    if i % num_instances == 0:
        plt.axvline(x=i - 0.5, color="blue", linestyle="-", linewidth=0.5)
        plt.axhline(y=i - 0.5, color="blue", linestyle="-", linewidth=0.5)


# Show the heatmap
plt.title("Trans-ReID trained from scratch")


