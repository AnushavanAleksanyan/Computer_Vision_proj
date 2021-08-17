from zipfile import ZipFile

file_name = "data.zip"

with ZipFile(file_name, 'r') as zip:
	zip.extractall()