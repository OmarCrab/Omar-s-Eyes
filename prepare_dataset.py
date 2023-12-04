import glob, os
from math import *
from tqdm import tqdm
import shutil

input_folders = [
    'output/yes',
    'output/no'
]

BASE_DIR_ABSOLUTE = 'dataset/'

OUT_DIR='./dataset/prepared'

OUT_TRAIN= OUT_DIR+'train/'
OUT_VAL=OUT_DIR+'test/'

coef = [80,20]
exceptions=['classes']

if int(coef[0])+int(coef[1])>100:
    print("Coeff can't exceed 100%")
    exit(1)

def chunker(seq,size):
    return(seq[pos:pos+size] for pos in range (0,len(seq),size))

print(f'Preparing image data by {coef[0]/coef[1]} rule')
print(f"Source folders: {len(input_folders)}")
print("Gathering data")

source = {}
for sf in input_folders:
    source.setdefault(sf,[])
    os.chdir(BASE_DIR_ABSOLUTE)
    os.chdir(sf)

    for filename in glob.glob("*.jpg"):
        source[sf].append(filename)

train={}
val={}
for sk, sv in source.items():
    chunks=10
    train_chunk=floor(chunks*(coef[0]/100))
    val_chunk=chunks-train_chunk

    train.setdefault(sk,[])
    val.setdefault(sk,[])
    for item in chunker(sv,chunks):
        train[sk].extend(item[0:train_chunk])
        val[sk].extend(item[train_chunk:])

train_sum = 0
val_sum = 0

for sk,sv in train.items():
    train_sum+=len(sv)

for sk,sv in val.items():
    val_sum+=len(sv)

print(f"\nOverral TRAIN images count {train_sum}")
print(f"Overral TEST images count {val_sum}")

os.chdir(BASE_DIR_ABSOLUTE)
print("\nCopying TRAIN source items to prepared folder . . . ")
for sk, sv in tqdm(train.items()):
    for item in tqdm(sv):
        imgfile_soutce = sk + item
        imgfile_dest = OUT_TRAIN+sk.split('/')[-2]+'/'

        os.makedirs(imgfile_dest,exist_ok=True)
        shutil.copyfile(imgfile_soutce,imgfile_dest+item)

os.chdir(BASE_DIR_ABSOLUTE)
print("\nCopying TEST source items to prepared folder . . . ")
for sk, sv in tqdm(train.items()):
    for item in tqdm(sv):
        imgfile_soutce = sk + item
        imgfile_dest = OUT_VAL+sk.split('/')[-2]+'/'

        os.makedirs(imgfile_dest,exist_ok=True)
        shutil.copyfile(imgfile_soutce,imgfile_dest+item)

print("\nDone!")