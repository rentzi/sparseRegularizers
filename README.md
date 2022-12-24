# sparseRegularizers
Generative model with different thresholding functions based on an upcoming paper


### Getting started
install jupyter, numpy, and matplotlib

follow the instructions and run `0 download preproces and whiten images` to download the natural images from the __Hateren Schaaf database__, to preprocess them, and to create and store input image patches for the generative algorithm

change the variables accordingly and run `1 Generative model with different regularizers` to process the input image patches with the generative model (create sparse codes and learn the overcomplete basis) with one of the four thresholding functions (ISTA, Hard thresholding, Half thresholding, CEL0)
