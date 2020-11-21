import os
import torch
import math

def writeManifest(path, totalRecords, batchSize, numBatches):
    f = open(path, 'w')
    s = "This directory is formatted for a BatchDataset.\n"
    s += "This file contains metadata.\n\n"
    s += "The total number of records is " + str(totalRecords) + "\n"
    s += "The batchsize is " + str(batchSize) + "\n"
    s += "The total number of batch fiies written is " + str(numBatches) + "\n"
    f.write(s)
    f.close()
    return True

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
    if writeManifest(os.path.join(dirname, 'manifest'), totalRecords, batchSize, numBatches) and verbose:
        print("Manifest written.")

    def _writeBatch(container, recordId):
        batchId = recordId // batchSize
        pathName = os.path.join(dirname, 'batch' + str(batchId))
        torch.save(container, pathName)
        if verbose:
            print("Records up to " + str(recordId + 1) +  " saved; " + str(batchId + 1) + " batches written.")
     
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

