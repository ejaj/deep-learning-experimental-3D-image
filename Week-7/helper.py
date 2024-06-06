import skimage
from matplotlib import pyplot as plt

datadir = 'data/glands/'
idx = 0


def nav_img(train_dir, idx):
    images = []
    labels = []
    for id in range(idx):
        img = f'{train_dir}{id:03d}.png'
        label = f'{train_dir}{id:03d}_anno.png'
        images.append(img)
        labels.append(label)
    print(images)
    print(labels)


nav_img(datadir + 'train/train_', 100)
label_im = skimage.io.imread('data/glands/train/train_000_anno.png')
plt.imshow(label_im)
plt.title("without margin")
plt.axis('off')  # Hide axes
plt.show()

with_label_im = skimage.io.imread('data/glands/train/train_000_anno.png')[20:-20, 20:-20] / 255
plt.imshow(with_label_im)
plt.title("with margin")
plt.axis('off')  # Hide axes
plt.show()
