# Final Year Project - Conor Meegan
## Project Title: Data augmentation using GANs for small image dataset
This is a UCD Computer Science final year project undertaken 
by Conor Meegan under the supervision of Prof. Soumyabrata Dev. 
The following project involves a Generative Adversarial Network 
(GAN). This is a type of neural network used to generate new 
photo-realistic images that are indistinguishable from the 
original images the network was trained on. The images on which 
this GAN will be trained are nighttime sky/cloud images. These 
images were captured using a ground-based sky camera. The dataset 
being used is the SWINSEG dataset which can be obtained through 
the following link: http://vintage.winklerbros.net/swinseg.html 

## Requirements
This project will use the deep learning API Keras which is 
written in Python. Keras will be running on top of the machine 
learning platform TensorFlow. TensorFlow is an end-to-end open 
source platform for machine learning. TenserFlow 2 is the 
version being used in this project. The current Python 
interpreter being used is version 3.7

## How To Run
<ol>
    <li>Download the SWINSEG dataset from the link provided above</li>
    <li>Clone the repository at:
    https://github.com/ConorMeegan/FinalYearProject.git</li>
    <li>Open up the GAN.py file</li>
    <li>Create four folders, one named 'images' another named 
    'saved_images', another named 'generated_images', and lastly one
    named 'binary_maps' at the same level as the GAN.py file</li>
    <li>Place the images downloaded from the link into the 'images' folder</li>
    <li>There will be three runs of the code performed. 
    The first to transform the image data. The second will be to train the GAN.
    The third will be to generate the accompanying binary maps for the images.</li>
    <li>Run the 'transform_images()' line of code in the main of the GAN.py file 
    to perform affine transformations on the images stored in the 'images' folder, 
    while commenting out the 'train_GAN()' and 'binary_maps()' lines of code</li>
    <li>The images should be saved in the 'saved_images' folder along with 
    the various transformations of the images</li>
    <li>Now comment out the 'transform_images()' line of code that create the 
    images for the 'saved_images' folder to save time while testing the running and
    training of the GAN</li>
    <li>Uncomment and run the 'train_GAN()' line of code in the main of the GAN.py 
    file for the number of epochs you wish to train the GAN for</li>
    <li>Once running, the summary of the Generator and Discriminator 
    networks will be printed out as well as the training progress for the number 
    of epochs chosen</li>
    <li>The images generated will be saved at certain intervals which can be changed
     by the user and stored in the 'generated_images' folder</li>
    <li>Now comment out the 'train_GAN()' line of code once the images have 
    been generated</li>
    <li>Finally, uncomment and run the 'binary_maps()' line of code after the GAN 
    has trained to generate the binary maps for the generated images. Set the 
    folder name to be the one with the images you wish to generate a binary map of. 
    Alter the interval in the for loop if necessary to fit the interval the GAN 
    saved images at.</li>
</ol> 