import torchvision
import torchvision.datasets

import util as u

originalDataDir = "CelebA/celeba/img_align_celeba/"
formattedDir = "sampleFormattedDir"

originalDataset = torchvision.datasets.ImageFolder(root = originalDataDir)

u.createFormattedDir(originalDataset, 'sampleFormattedDir', batchSize = 128, cleanDir = True)


