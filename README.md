# Refining Coarse Annotations on Single Whole-Slide Image
Detailed and exhausitive annotations on whole-slide images (WSI) is extremely labor-intensive and time-consuming. In this repository, we provide implementation of two methods -- (1) Deep k-NN (DkNN); (2) Label Cleaning Multiple Instance Learning -- for refining these coarse annotations, and producing a more accurate version of annotations. The figure below shows an example of the coarse annotations and the refined annotations produced by one of our method (LC-MIL). 
![image.png](https://boostnote.io/api/teams/m3RRYyWgW/files/971384651b34f3d66ffbcbd396653b8bd7fc9fc9532a5e22d21f2cbf3648d1d7-image.png)
Noticeably, although both methods are machine learning based, the refinement can be conducted on each single slide, and no externel data is needed.

## Dataset
- [CAMELYON16](https://camelyon16.grand-challenge.org/)
- [PAIP2019](https://paip2019.grand-challenge.org/)
- [PAIP2020](https://paip2020.grand-challenge.org/Dataset/)
- One WSI for test: https://drive.google.com/file/d/1ArNlIWZtqfHHb_9S85iIocmSVaHgCQBI/view?usp=sharing

## Usage
### DkNN
#### Training
```
python train_model.py 

positional arguments:
  slide_root            The location of the whole-slide image.
  slide_ID              Name of the whole-slide image.
  slide_format          Dataformat the whole-slide image. Permitted format can
                        be `.svs`, `,ndpi`, and `.tif`.
  ca_path               The path to the coarse annotations. File format should
                        be `.sav`
  model_save_root       Where model will be saved
  
optional arguments:
  -h, --help            show this help message and exit
  --remove_blank REMOVE_BLANK
                        How to remove blank regions (i.e. identify tissue
                        regions) of WSI. We provide three functions: 0:
                        convert to HSV and then OTSU threshold on H and S
                        channel; 1: apply [235, 210, 235] on RGB channel,
                        respecitively; 2: convert to gray image, then OTSU
                        threshold. Default is 0. For new dataset, the user is
                        encouraged to write customed function
  --focal_loss FOCAL_LOSS
                        Whether or not to use focal loss (True: using focal
                        loss; Flase: using cross entropy), default is CE
  --patch_shape PATCH_SHAPE
                        Patch shape(size), default is 256
  --unit UNIT           Samllest unit when cropping patches, default is 256
  --gpu GPU             gpu
  --lr LR               Initial Learning rate, default is 0.00005
  --step_size STEP_SIZE
                        Step size when decay learning rate, default is 1
  --reg REG             Reg,default is 10e-5
```
#### Applying/Inference
```
python apply_model.py 

positional arguments:
  slide_root            The location of the whole-slide image.
  slide_ID              Name of the whole-slide image.
  slide_format          Dataformat the whole-slide image. Permitted format can
                        be `.svs`, `,ndpi`, and `.tif`.
  ca_path               The path to the coarse annotations. File format should
                        be `.sav`
  model_dir             Where to load the model (to conduct feature
                        extraction)
  feature_save_root     Where the mapped features will be saved
  knn_save_root         Where the KNN results (distance and index) will be
                        saved
  heatmap_save_root     Where the predicted heatmap will be saved

optional arguments:
  -h, --help            show this help message and exit
  --remove_blank REMOVE_BLANK
                        How to remove blank regions (i.e. identify tissue
                        regions) of WSI. We provide three functions: 0:
                        convert to HSV and then OTSU threshold on H and S
                        channel; 1: apply [235, 210, 235] on RGB channel,
                        respecitively; 2: convert to gray image, then OTSU
                        threshold. Default is 0. For new dataset, the user is
                        encouraged to write customed function
  --focal_loss FOCAL_LOSS
                        Whether or not to use focal loss (True: using focal
                        loss; Flase: using cross entropy), default is CE
  --patch_shape PATCH_SHAPE
                        Patch shape(size), default is 256
  --unit UNIT           Samllest unit when cropping patches, default is 256
  --gpu GPU             gpu

```
#### Template command
```
cd DkNN
python train_model.py ../Data test_016 .tif ../coarse_annotations.sav . 
python apply_model.py ../Data test_016 .tif ../coarse_annotations.sav model_test_016.pth . . . 
```
We can not actually upload our test WSI, `test_016.tif` to the `/Data` in this repository due to the space limit of Github, but you can found it in the [google drive](https://drive.google.com/file/d/1ArNlIWZtqfHHb_9S85iIocmSVaHgCQBI/view?usp=sharing)

### LC-MIL
#### Training
```
cd LC_MIL
python train_model.py 

positional arguments:
  slide_root            The location of the whole-slide image.
  slide_ID              Name of the whole-slide image.
  slide_format          Dataformat the whole-slide image. Permitted format can
                        be `.svs`, `,ndpi`, and `.tif`.
  ca_path               The path to the coarse annotations. File format should
                        be `.sav`
  model_save_root       Where model will be saved

optional arguments:
  -h, --help            show this help message and exit
  --remove_blank REMOVE_BLANK
                        How to remove blank regions (i.e. identify tissue
                        regions) of WSI. We provide three functions: 0:
                        convert to HSV and then OTSU threshold on H and S
                        channel; 1: apply [235, 210, 235] on RGB channel,
                        respecitively; 2: convert to gray image, then OTSU
                        threshold. Default is 0. For new dataset, the user is
                        encouraged to write customed function
  --length_bag_mean LENGTH_BAG_MEAN
                        Average length of bag (Binomial distribution),default
                        = 10
  --num_bags NUM_BAGS   Number of bags to train,default = 1000
  --focal_loss FOCAL_LOSS
                        Whether or not to use focal loss (True: using focal
                        loss; Flase: using cross entropy), default is FL
  --patch_shape PATCH_SHAPE
                        Patch shape(size), default is 256
  --unit UNIT           Samllest unit when cropping patches, default is 256
  --gpu GPU             gpu
  --lr LR               Initial Learning rate, default is 0.00005
  --step_size STEP_SIZE
                        Step size when decay learning rate, default is 1
  --reg REG             Reg,default is 10e-5
```
#### Applying/Inference
```
python apply_model.py 

positional arguments:
  slide_root            The location of the whole-slide image.
  slide_ID              Name of the whole-slide image.
  slide_format          Dataformat the whole-slide image. Permitted format can
                        be `.svs`, `,ndpi`, and `.tif`.
  model_dir             The path to the MIL model
  heatmap_save_root     Where the predicted heatmap will be saved

optional arguments:
  -h, --help            show this help message and exit
  --remove_blank REMOVE_BLANK
                        How to remove blank regions (i.e. identify tissue
                        regions) of WSI. We provide three functions: 0:
                        convert to HSV and then OTSU threshold on H and S
                        channel; 1: apply [235, 210, 235] on RGB channel,
                        respecitively; 2: convert to gray image, then OTSU
                        threshold. Default is 0. For new dataset, the user is
                        encouraged to write customed function
  --length_bag_mean LENGTH_BAG_MEAN
                        Average length of bag (Binomial distribution),default
                        = 10
  --num_bags NUM_BAGS   Number of bags to train,default = 1000
  --focal_loss FOCAL_LOSS
                        Whether or not to use focal loss (True: using focal
                        loss; Flase: using cross entropy), default is FL
  --patch_shape PATCH_SHAPE
                        Patch shape(size), default is 256
  --unit UNIT           Samllest unit when cropping patches, default is 256
  --gpu GPU             gpu
```
#### Template command
```
cd LC_MIL
python train_model.py ../Data test_016 .tif ../coarse_annotations.sav . 
python apply_model.py ../Data test_016 .tif model_test_016.pth . . . 
```
We can not actually upload our test WSI, `test_016.tif` to the `/Data` in this repository due to the space limit of Github, but you can found it in the [google drive](https://drive.google.com/file/d/1ArNlIWZtqfHHb_9S85iIocmSVaHgCQBI/view?usp=sharing)

### Post-processing
Post-processing procedure and the illustration of results can be found in `post_process.ipynb`.

## Publication
