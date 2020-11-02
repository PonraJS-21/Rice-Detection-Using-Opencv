## Rice detection

### Step -1: Create Dateset

Run keras_datagenreator_data_augmentation.ipynb file and check <i><b>all</b></i> folder for dataset, Now move these datasets to create_classifier/dataset_image folder

### Step -2: Create Model

Now run the train-VGG16.py file in create_classifier folder, Once it executed check for .pick, .JSON, .h5 files in the same folder, now move these three files to main_dir/models(create if not exist)


#### Now run rice_detection.py to detect rice grains.