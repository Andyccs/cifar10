import os

dataset_dir = 'dataset/'

files = [
  ('sampleSubmission.csv', 3188904), 
  ('test.7z', 639374249), 
  ('train.7z', 109723070), 
  ('trainLabels.csv', 588903)
]


for f in files:
  statinfo = os.stat(dataset_dir + f[0])
  if statinfo.st_size == f[1]:
    print 'Found and verified: ', f[0]