import os
import torch
import math

def writeManifest(dirname, totalRecords, batchSize, numBatches):
    f = open(os.path.join(dirname, 'manifest'), 'w')
    s = "This directory is formatted for a BatchDataset.\n"
    s += "This file contains metadata.\n\n"
    s += "The total number of records is " + str(totalRecords) + "\n"
    s += "The batchsize is " + str(batchSize) + "\n"
    s += "The total number of batch fiies written is " + str(numBatches) + "\n"
    f.write(s)
    f.close()
    return True


def parseManifest(dirname):
    f = open(os.path.join(dirname, 'manifest'), 'r') # Put this in a try-except
    s = f.read()
    f.close()
    l = s.split('\n')[3:]
    totalRecords = int(l[0].split(' ')[-1])
    batchSize = int(l[1].split(' ')[-1])
    numBatches = int(l[2].split(' ')[-1])
    return totalRecords, batchSize, numBatches


def verifyDirEmpty(dirname, cleanDir=False):
    if len(os.listdir(dirname)) > 0:
        if cleanDir:
            for fname in os.listdir(dirname):
                os.remove(os.path.join(dirname, fname))
        else:
            raise OSError("Directory '" + dirname  + "' not empty; choose a new directory or set cleanDir=True.")
    return True 


def createFormattedDir(source, dirname, batchSize = 128, cleanDir=False, verbose=True):
    """Given a source that supports __len__ and __getitem__, 
and a directory that's empty, create a formatted directory that's 
then used by BatchDataset. This only needs to be run once, so it's a utility. 
Adjust batchSize so two batch leaves enough (CPU) RAM for normal processes, 
but is as large as possible. BathDataset sometimes loads two batches, 
to avoid waiting on a disk read."""
    if verifyDirEmpty(dirname, cleanDir=cleanDir) and verbose:
        print("Directory '" + dirname + "' is empty.")
    totalRecords = len(source)
    numBatches = math.ceil(totalRecords / batchSize)
    if writeManifest(dirname, totalRecords, batchSize, numBatches) and verbose:
        print("Manifest written.")

    def _writeBatch(container, recordId):
        batchId = recordId // batchSize
        pathName = os.path.join(dirname, 'batch' + str(batchId))
        torch.save(container, pathName)
        if verbose:
            print("Records up to " + str(recordId + 1) +  " saved; " + str(batchId + 1) + " batches written.")

    # Main loop to store all the records.
    # By default, python list are the batches, but the BatchDataset class will work with any container.
    res = []
    for i in range(totalRecords):
        res.append(source[i])
        if (i+1) % batchSize == 0:
            _writeBatch(res, i)
            res = []

    if len(res) > 0: # Take care of stragglers.
        _writeBatch(res, i)
        res = []
   
    print("Finished! Directory '" + dirname + "' is ready to be used for a BatchDataset.")
    return True 

