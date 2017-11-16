# DeepCell

## Hardware
I have been running DeepCell on a Puget systems workstation that has a 6 core Intel Xeon processor and 2 Nvidia GTX980 graphics cards. We also run DeepCell on Stanford's Sherlock cluster that uses Nvidia GTX Titan Black graphics cards. We have not tested our code on other setups, but any computer that has a CUDA and cuDNN compatible video card should be fine.

## Installation
Run the following commands to install the required dependencies

* pip install numpy
* pip install scipy
* pip install scikit-learn matplotlib palettable scipy libtiff

The latest development version of scikit-image will need to be installed. It's github repository can be found at https://github.com/scikit-image/scikit-image. The theano package will also need to be installed. Further details about installing theano can be found at http://deeplearning.net/software/theano/install.html. We use the bleeding edge installation of Theano, as it contains a pooling function with variable strides which is necessary for rapid execution of this software.

## Brief workflow overview
DeepCell contains three main networks in three different folders
* Bacteria net segments phase images of E. coli. 
* Nuclei net segments fluorescent images of mammalian cell nuclei
* Feature net was developed to both segment phase images of mammalian cells. With appropriate training data, it can also segment cells and determine their cell type

## Making training data sets
We use ImageJ to construct our training data sets. We've found the most success in using phase microscopy images in conjunction with a WGA stain (or any other membrane/cytoplasmic marker) to identify the boundaries of each cell in these images. To make a training data set, we perform the following steps.
* We first create a folder called "training_data" and for each image we plan to manually segment for the training dataset create a subfolder within this folder and move the images (both phase and nuclear marker) to a the subfolder. We use different subfolder names for each image, but the image names are the same (for instance "phase.png" and "dapi.png").
* We first outline each cell in ImageJ using the freehand selection tool and the ROI manager. Once all of the cells are selected, we then create a new image in ImageJ (with the same dimensions) and use the draw option in ImageJ to draw the cell outlines into the new image. This new image is then saved as "feature_0.png" (for feature net) or "edge.png" (for bacteria or nuclei net) and is the location of all of the "edge" pixels in the training image. 
* The next step is to create an equivalent image for the "interior" pixels. We then repeat the preceeding two steps (creating a new image and drawing the cell contours) and use the flood fill tool to fill in each individual cell. We then subtract the edge outline image from this new image and save the result as "feature_1.png" (feature net) or "interior.png" (bacteria or nuclei net).
* For making training datasets for semantic segmentation, we perform the above steps but save each cell type as its own "feature_x.png" file. For instance, for the MCF10A and NIH-3T3 semantic segmentation datasets, the MCF10A cell interior was saved as "feature_2.png" while the NIH-3T3 cell interior was saved as "feature_1.png." Because no NIH-3T3 cells were present in the training image of MCF10A, the "feature_1.png" file was just an image of 0's (and vice versa for NIH-3T3 cells).
* Once a training image is made, we then run the "make_training_data.py" script to construct an npz file that contains the training data and can be loaded by the subsequent scripts. Before running, we first need to direct the script file to the location of the manually segmented images. This is done by changing the "direc_name" variable to the name of the folder containing the training data, and changing the "training_direcs" variable to a list containing all of the images to be included in the training data set.
* To limit the total size of the training data set, the max_training_examples field can be changed to the maximum number of training examples desired.
* The name of the file containing the training dataset is specified in the variable "file_name_save." 

Once the "make_training_data.py" file has been modified, run the command >> python make_training_data.py

## Training conv-nets
The workflow for training each network is similar. Each folder contains a CNN_layers.py file that contains all of the functions used to construct a convolutional neural network (conv-net). Training data is constructed by running the make_training_data.py file on the manually segmented images as detailed above. The network architecture is specified in the CNN_(network type)_net_(receptive field size).py file. Training is performed by running the train_CNN_(network type).py file. On a GTX 980 GPU, we've found that it typically takes 5-10 hours to train a conv-net. Because we typically train 4 networks at a time, training typically takes 1-2 days. 

To train a conv-net on a training dataset, we first have to modify the "train_CNN_(network type).py" file to direct it to the appropriate training dataset and specify where the conv-net parameters are going to be saved.
* The variable "direc_name" needs to be changed to the directory containing the training data. Similarly, the variable "training_name" needs to be changed to the name of the .npz file containing the training data.
* The variables "date" and "dataset_name" specify the file name where the network parameters are saved while the variable "direc_save" specifies the directory where the network parameter file is saved.

Once the variables are specified, run the command >> python train_CNN_(network type).py

## Executing conv-nets
The pipeline.py file is used to run a trained conv-net on every image file within a folder. These images are usually time lapse microscopy images acquired from MicroManager and usually have a name structure of "img_(frame number)_(Channel name)." A few things need to be specified in order to process microscope images.
* The folder to be processed is specified in the "direc_name" variable. The images to be processed should be stored in a subfolder called "RawImages." 
* pipeline.py creates several folders to save the processed images. These folders are "Align", "Cropped", "Output", "Nuclei", and "Masks" 
* The different imaging channels are specified in the channel_names variable - these channels must appear in the image names for the images to be processed. The phase channel should be the first channel in the list, followed by the nuclear marker channel.
* The networks to be run should be specified using the "cnn_date" and "cnn_dataset_name" variables. The "cnn_parameter_location" variable should be modified so it points to the directory containing the conv-net parameter file.
* There are two pipeline.py files in the feature_net folder - pipeline.py and pipeline_semantic.py. The latter is used for semantic segmentation.

There are 6 trained networks available in the download. They include
* feature_net/trained_networks/DVV_10202015_training_features_61x61. These conv-nets perform semantic segmentation of NIH-3T3 and MCF10A cells.
* feature_net/trained_networks/DVV_11022015_HeLa_training_features_61x61. These conv-nets segment HeLa cells.
* feature_net/trained_networks/DVV_11072015_3T3_training_features_61x61. These conv-nets segment 3T3 cells.
* feature_net/trained_networks/DVV_11072015_MCF10A_training_features_61x61. These conv-nets segment MCF10A cells.
* nuclei_net/trained_networks/DVV_01062016_training_nuclear. These conv-nets segment fluorescent images of nuclei.
* bacteria_net/trained_networks/bacteria_net_03242015_21x21_thick3. These conv-nets segment E. coli cells.

The pipeline file has 4 components
* Making directories. The program constructs directories to store images at different steps of the processing.
* Aligning images - this is optional. Sometimes there is drift in movies - activating this component aligns all of the images using a cross correlation based method. The images are then cropped and saved in the "Align" subdirectory. To activate this, make sure the line of code under the "Align" heading is uncommented and change the "data_location" field in the run_cnn function to "align_dir". To keep it inactive, make sure the "data_location" fields are set to "image_dir"
* Running the conv-net. The conv-net to be run needs to be specified in the "cnn_date", "cnn_dataset_name", and "cnn_paramater_location" variables as these variables direct the pipeline.py file to the conv-net parameters that were produced from the training procedure. This section also runs several conv-nets in sequence (if multiple conv-nets were trained in succession in the train_CNN file) and then averages the results. To change how many models are averaged, change the "number_of_models" variable to the specified number.
* Saving the conv-net output. The output of the conv-nets are saved in the "Output" folder.

When the variables are specified, run the command >> python pipeline.py

## Downstream processing and segmentation refinement
The downstream.py file performs the segmentation refinement using active contours and produces the final segmentation masks. Before running this script, change the "direc_name" variable to the directory to be processed. 

Once the variable is specified, run the command >> python downstream.py


