#!/bin/bash
# Downloads Mini-WebVision (the first 50 Google-image classes of WebVision) and
# re-arranges it into an ImageNet-style train/<class>/*.jpg, val/<class>/*.jpg
# layout so it can be read by `dataloader/miniwebvision.py` (torchvision
# ImageFolder convention). Run this script from inside data/miniwebvision/.
# Requires ~30GB of free disk space and `wget`.

### Download files
wget https://data.vision.ee.ethz.ch/cvl/webvision/google_resized_256.tar
wget https://data.vision.ee.ethz.ch/cvl/webvision/val_images_256.tar
wget https://data.vision.ee.ethz.ch/cvl/webvision/info.tar

### Make directory and move tar files
mkdir ./train
mkdir ./val
mv ./google_resized_256.tar ./train/
mv ./val_images_256.tar ./val/

### Move and uncompress the train files
cd ./train
tar -xf google_resized_256.tar
cd ../

### Move and uncompress the val files
cd ./val
tar -xf val_images_256.tar
cd ..

### Move and uncompress the info files
tar -xf info.tar

### Make imagenet_folder_name
python build_imagenet_folder_map.py

### Copy train images to respective folder
echo "----------------------------------------------------------------"
echo "Creating directory structure similar to ImageNet for training dataset"
echo "----------------------------------------------------------------"
input="./info/imagenet_folder_name.txt"
path_folder="./train"
while IFS= read -r line
do
    web_folder_name=$(echo "$line" | cut -c 1-5)  # Extract first 5 characters
    img_folder_name=$(echo "$line" | cut -c 7-) 
    mkdir -p $path_folder/$img_folder_name
    mv $path_folder/google/$web_folder_name/* $path_folder/$img_folder_name/
    rm -rf $path_folder/$web_folder_name
done < "$input"

### Copy val images to respective folder
echo "----------------------------------------------------------------"
echo "Creating directory structure similar to ImageNet for val dataset"
echo "----------------------------------------------------------------"
input="./info/val_imagenet_class.txt"
path_folder="./val"
while IFS= read -r line
do
    imagenet_name=$(echo "$line" | cut -c 1-13)  # Extract the first 13 characters
    img_folder_name=$(echo "$line" | cut -c 15-)  # Extract everything from the
    mkdir -p $path_folder/$img_folder_name
    mv $path_folder/val_images_256/$imagenet_name $path_folder/$img_folder_name/
done < "$input"



# remove files 
echo "----------------------------------------------------------------"
echo "Removing Redundant files."
echo "----------------------------------------------------------------"
rm -rf ./info.tar 
rm -rf ./val/val_images_256.tar 
rm -r val/val_images_256/
rm -rf ./train/google_resized_256.tar
rm -r train/google/



echo "----------------------------------------------------------------"
echo "Mini-WebVision Dataset Processed!"
echo "----------------------------------------------------------------"
