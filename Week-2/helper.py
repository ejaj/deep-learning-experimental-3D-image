import os

IN_DIR_DATA = 'data/CUB_200_2011/'
path_to_splits = os.path.join(IN_DIR_DATA, 'train_test_split.txt')
print(path_to_splits)
indices_to_use = list()

with open(path_to_splits, 'r') as in_file:
    for line in in_file:
        idx, use_train = line.strip('\n').split(' ', 2)
        if bool(int(use_train)):
            indices_to_use.append(int(idx))
# print(indices_to_use)

# obtain filenames of images
path_to_index = os.path.join(IN_DIR_DATA, 'images.txt')
filenames_to_use = set()
with open(path_to_index, 'r') as in_file:
    for line in in_file:
        idx, filename = line.strip('\n').split(' ', 2)
        if int(idx) in indices_to_use:
            filenames_to_use.add(filename)
print(filenames_to_use)

