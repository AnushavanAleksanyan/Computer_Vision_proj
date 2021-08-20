import numpy as np
import pandas as pd
import os
import shutil
from zipfile import ZipFile
from glob import glob

def acc():
	ans = input("Type Y to unzip the archive: ")
	if ans.lower() == "y":
		return True


archive = "initial_data/500MB.zip"
"""
if acc():
	with ZipFile(archive, 'r') as zip:
	 	zip.extract("500MB/labels.csv", path ="data")

if acc():
	with ZipFile(archive) as zip_file:
		zip_file.extractall("initial_data/")
"""

df = pd.read_csv("data/500MB/labels.csv")

# print(df.head(50))
# print(df.columns)

folder_path = "data/train/"
folder_list = glob("data/train/*/")
f_names = ["Black", "Blue", "Green", "Pink", "Red", "White"]

# for index, row in df.iterrows():
# 	print (row['id'], row['Color'])
files=os.listdir("initial_data/500MB/images/2302883/")
"""

for fname in files:
	shutil.copy2(os.path.join("initial_data/500MB/images/2302883/",fname),"data/train/white/")
"""

def move(id, color):
	for address, dirs, files in os.walk("initial_data/500MB/images/"):
		for dir in dirs:
			files2 = os.listdir(f"initial_data/500MB/images/{id}/")
			for fname in files2:
				shutil.copy2(os.path.join(f"initial_data/500MB/images/{id}/",fname),f"data/train/{color}/")
			os.rename(f"data/train/{color}/{fname}",f"data/train/{color}/{fname} - {dir}")

for index, row in df.iterrows():
	if not row['Color'] in f_names:
		move(row['id'], "other")
		print(row['id'], "other")
	else:
		move(row['id'], row["Color"])
		print(row['id'], row["Color"])
