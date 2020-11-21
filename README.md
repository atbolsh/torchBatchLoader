An attempt to create a good batched Dataset object for torch, so that the number of disk reads when working with something like CelebA is minimized.

Hopefully, this will dramatically improve speed, especially in environments with remote disks, like Google Colab.


