import matplotlib.pyplot as plt 
import cv2 
IDX = 10
img = cv2.imread(f"chairs22/{IDX}.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
depth = cv2.imread(f"chairs22/{IDX}_d.jpg", cv2.IMREAD_GRAYSCALE)
fig, axs = plt.subplots(1, 2)
axs[0].imshow(img)
axs[1].imshow(depth, cmap="viridis")
# axs[1].colorbar()
plt.savefig("vis.jpg")
