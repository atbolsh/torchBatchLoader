import torchvision
import torchvision.datasets

import util as u
import batchDataset as bd

import time

originalDataDir = "CelebA/celeba/img_align_celeba/"
formattedDir = "sampleFormattedDir"

originalDataset = torchvision.datasets.ImageFolder(root = originalDataDir)
batchDataset = bd.BatchDatasetSimple(formattedDir)


print("Scrolling through original dataset")
t1 = time.time()
for i in range(len(originalDataset)):
    img = originalDataset[i]
    if (i + 1) % 100 == 0:
        print("Up to image " + str(i))
t2 = time.time()

print("Time to scroll through original dataset is " + str(t2 - t1) + " seconds.")
print("Scrolling through batched dataset")

t3 = time.time()
for i in range(len(batchDataset)):
    img = batchDataset[i]
    if (i + 1) % 100 == 0:
        print("Up to image " + str(i))
t4 = time.time()

print("Time to scroll through batched dataset is " + str(t4 - t3) + " seconds.")


print("To recapitulate:")

print("Time to scroll through original dataset is " + str(t2 - t1) + " seconds.")
print("Time to scroll through batched dataset is " + str(t4 - t3) + " seconds.")
print("End of speed test.")
