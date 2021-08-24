import numpy as np
import pandas as pd
import os
import shutil
from zipfile import ZipFile
import pathlib
import random
# import dircache


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
		img_count = 0
		for file in os.listdir(f"data/train/{folder}/"):
			img_count += 1
			if img_count%10 ==0:
				filename = random.choice(os.listdir(f"data/train/{folder}/"))
				shutil.move(os.path.join(f"data/train/{folder}/",filename),os.path.join(f"data/test/{folder}/",filename))



devide_test()

