import numpy as np
import pandas as pd
import os
import shutil
from zipfile import ZipFile
import pathlib
from glob2 import glob

def acc():
	ans = input("Type 'Y' to unzip the archive: ")
	if ans.lower() == "y":
		return True


archive = "500MB.zip"

if acc():
	#os.mkdir("initial_data")
	with ZipFile(archive, 'r') as zip:
	 	zip.extractall()
	os.rename(os.path.abspath("500MB"), os.path.abspath("initial_data"))

def create_data_folders():
	os.mkdir(os.path.abspath("data"))
	os.mkdir(os.path.abspath("data/train"))
	os.mkdir(os.path.abspath("data/test"))
	os.mkdir(os.path.abspath("data/validation"))
	return("created")

# create_data_folders()


def create_folders_for_features():
	df = pd.read_csv("initial_data/labels.csv")
	colors = df["Color"].unique()
	for color in colors:
		os.mkdir(os.path.abspath("data/train/"+color))
		os.mkdir(os.path.abspath("data/test/"+color))
		# os.mkdir(os.path.abspath("data/validation/"+color))
	return("made!")

# create_folders_for_features()


df = pd.read_csv("initial_data/labels.csv")

folders = [ name for name in os.listdir("initial_data/images/") if os.path.isdir(os.path.join("initial_data/images/", name)) ]

def move_images():
	for id, color in df[["id","Color"]].values:
		files = [f for f in os.listdir(f"initial_data/images/{id}/") if os.path.isfile(os.path.join(f"initial_data/images/{id}/", f))]
		for file in files:
			print(file, id, color)
			shutil.copy2(os.path.join(f"initial_data/images/{id}/",file),f"data/train/{color}/{str(id)+file}")


# move_images()


def devide_test():
	for folder in os.listdir(f"data/train/"):
		for file in os.listdir(f"data/train/{folder}/"):
			print(file)


devide_test()












"""
df = pd.read_csv("data/500MB/labels.csv")



def move(id, color):
	for address, dirs, files in os.walk("initial_data/500MB/images/"):
		for dir in dirs:
			files2 = os.listdir(f"initial_data/500MB/images/{id}/")
			for fname in files2:
				shutil.copy2(os.path.join(f"initial_data/500MB/images/{id}/",fname),f"data/train/{color}/")
		rename_photos(color)


def arrange_img(var, id, color, file):
	shutil.copy2(f"initial_data/500MB/images/{id}/{file}",f"data/{var}/{color}/{str(id)+file}")


if __name__ == "__main__":
	filelist = []
	folderlist = []
	i=0
	for root,dirs,files in os.walk(f"initial_data/500MB/images/"):

		print(os.listdir(root))
			# for file in files:
			# 	print(file)
	# 		filelist.append(file)
	# 		folderlist.append(root)
	# # print(len(filelist))
	# print(len(folderlist))
	#print(dirlist)
	# fold = iter(dirlist)
	# if i%10 == 0:
	# 	var = "test"
	# else:
	# 	var = "train"
	# print(df[df['id']==next(fold)])
	# if not df[df['id']==next(fold)]["Color"] in f_names:
	# 	#arrange_img(var, folderlist[i], "other",filelist[i])
	# 	print(var, folderlist[i], "other", filelist[i])
	# else:
	# 	#arrange_img(var, folderlist[i], row["Color"], filelist[i])
	# 	print(var, folderlist[i], df[df['id']==next(fold)]["Color"], filelist[i])
	# i += 1

# result = df[df['id']==2462198]["Color"]
# print(result)

# for fname in filelist:
"""