# sparseRegularizers
Generative model with different thresholding functions based on an upcoming paper


### Getting started
install jupyter, numpy, and matplotlib

follow the instructions and run `0 download preproces and whiten images` to download the natural images from the __Hateren Schaaf database__, to preprocess them, and to create and store input image patches for the generative algorithm

change the variables accordingly and run `1 Generative model with different regularizers` to process the input image patches with the generative model (create sparse codes and learn the overcomplete basis) with one of the four thresholding functions (ISTA, Hard thresholding, Half thresholding, CEL0)

This implementation of the generative models uses a learning model where the learning rates decay with iterations based on a time-based decay schedule, with decay rates for both learning rates being their initial values divided by 50. We used this implementation for CEL0 for dictionary sizes greater than 500 units. For the other methods the learning rates were the same across iterations. If you want to run the later implementation set the vectors `lr_rV` and `lrPhiV` in `generativeModels.py` to fixed values
