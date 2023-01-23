import glob 
import os
import shutil
from pathlib import Path

myDir = 'A:\\Research\\Training_data\\run_1_gan\\coordinates\\'
tempDir = '9'
myDir = myDir + tempDir + '\\'

### 
count = 0 
for infile in sorted(glob.glob(myDir + '*.txt')):
    name = Path(infile).stem
    name = int(name)

    # Renaming 
    os.rename(infile, myDir + str(count) + '.txt')
    count = count + 1
