import numpy as np
import pandas as pd
import os
import shutil
from zipfile import ZipFile
import pathlib

def acc():
	ans = input("Type 'Y' to unzip the archive: ")
	if ans.lower() == "y":
		return True

def rename_photos(fold):
	path = pathlib.Path('.')/f"data/train/{fold}/"
	for folder in path.iterdir():
		if folder.is_dir():
			counter = 0
			for file in folder.iterdir():
				if file.is_file():
					new_file = folder.name + "_" + str(counter) + file.suffix
					file.rename(path / folder.name / new_file)
					counter += 1


archive = "initial_data/500MB.zip"

if acc():
	with ZipFile(archive, 'r') as zip:
	 	zip.extract("500MB/labels.csv", path ="data")

	with ZipFile(archive) as zip_file:
		zip_file.extractall("initial_data/")


df = pd.read_csv("data/500MB/labels.csv")

# print(df.head(50))
# print(df.columns)


f_names = ["Black", "Blue", "Green", "Pink", "Red", "White"]

def move(id, color):
	i=0
	for address, dirs, files in os.walk("initial_data/500MB/images/"):
		i+=1
		for dir in dirs:
			if i % 10 != 0:
				dest = "train"
			else:
				dest = "test"
			print(dest)
			files2 = os.listdir(f"initial_data/500MB/images/{id}/")
			for fname in files2:
				shutil.copy2(os.path.join(f"initial_data/500MB/images/{id}/",fname),f"data/{dest}/{color}/")
		print (i)
		rename_photos(color)
			
	


for index, row in df.iterrows():
	if not row['Color'] in f_names:
		move(row['id'], "other")
		print(row['id'], "other")
	else:
		move(row['id'], row["Color"])
		print(row['id'], row["Color"])

