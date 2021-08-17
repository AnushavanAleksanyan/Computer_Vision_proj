import numpy as np
import pandas as pd
from zipfile import ZipFile
import glob

archive_name = "initial_data/500MB.zip"

with ZipFile(archive_name, 'r') as zip:
 	zip.extract("500MB/labels.csv", path ="data")

df = pd.read_csv("data/500MB/labels.csv")

# print(df.head(50))
# print(df.columns)

folder  = df["id"].iloc[0]
color = df["Color"].iloc[0]
print(glob.glob("data"))

directories = glob.glob("data/test/*/")
print(directories)