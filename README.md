## Rice detection

### Step -1: Create Dateset

<b> Run keras_datagenreator_data_augmentation.ipynb file and check all folder for dataset, Now move these datasets to create_classifier/dataset_image folder</b>

### Step -2: Create Model

<b> Now run the train-VGG16.py file in create_classifier folder, Once it executed check for .pick, .JSON, .h5 files in the same folder, now move these three files to main_dir/models(create if not exist) </b>


#### Now run rice_detection.py to detect rice grains.