---
title: "Bayesian Transfer-Learning Feature Extraction"
excerpt: "Detecting Pneumonia in chest X-rays.<br/><img src='/images/pneumonia_cxr_nn/pneumonia_project_cover.png'>"
collection: portfolio
---
{%include base_path%}

Pneumonia is an infection wherein one or both of the lungs are 
incapacitated by fluids in the air-sacs of the lung. Pneumonia accounts 
for about 2 million deaths yearly in children under the age of 5. Pneumonia is usually 
diagnosed by examination of chest X-rays wherein Pneumonia appears as cloudiness 
within the lungs.

<!-- ![sidebyside xrays or normal/pneumonia chests]('/pneumonia_cxr_nn/normal_pneumonia_cxr.png' "normal chest x-ray next to a chest x-ray exihiting pneumonia") -->
<center>
<figure>
    <img width="100%" src="/images/pneumonia_cxr_nn/normal_pneumonia_cxr.png" alt='normal chest x-ray sitiated next to a chest x-ray exhibiting pneumonia'/>
</figure>
</center>

Automated medical image diagnoses streamlines the usage of human medical 
resources. In this case a radiologist who visually inspects the X-rays 
to see if there is the presence of pneumonia. In the visual inspection case 
this radiologist would give equal attention to each image they receive, and 
issuing a diagnosis or lack thereof. It's more important for the 
radiologist to give their attention to the possible pneumonia x-rays. 
An AI system that could pre-classify chest X-rays could expedite the 
diagnostic workflow of the radiologist and reduce times between 
examination, diagnosis, and more importantly, treatment. 

The dataset I use in this project comes from the [Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images](https://data.mendeley.com/datasets/rscbjbr9sj/3)
which looked at using transfer-learning in order to explore the feasibility
of early detection of specific ailments that are typically diagnosed from
visual examination of X-rays. I quote the two main sections of text that best
describe the Pneumonia dataset, as well as the methods of labeling that were used in
the black bordered box below.

<div>
    <div style="border: 8px solid black; display: inline-block; padding: 4px;">

<b>Description:</b> "We collected and labeled a total of 5,232 chest X-ray images
from children, including 3,883 characterized as depicting pneumonia 
(2,538 bacterial and 1,345 viral) and 1,349 normal, from
a total of 5,856 patients to train the AI system. The model was
then tested with 234 normal images and 390 pneumonia images
(242 bacterial and 148 viral) from 624 patients." 

<b>Labeling:</b> "For the analysis of chest X-ray images, all chest radiographs were 
initially screened for quality control by removing all low quality or
unreadable scans. The diagnoses for the images were then graded by two 
expert physicians before being cleared for training the AI system. In 
order to account for any grading errors, the evaluation set was also 
checked by a third expert."
    </div>
</div>


A neural network classifier that only outputs a ‘positive’ or 
‘negative’ diagnosis is not gonna cut it. It’s important to understand 
here that non-Bayesian neural networks output classifications that do 
not take into account either the model uncertainty (otherwise called 
*epistemic uncertainty*), or the uncertainty in the image data itself 
(called *aleatoric uncertainty*). 

I decided to employ transfer-learning to perform feature extraction on
the image data. Transfer-learning for feature extraction consists of 
using a pre-trained convolutional neural network in order to take 
advantage of the larger architecture's and training datasets used in
many pre-trained models. The output of this pre-trained model is then
flattened and fed through dense layers in order to perform the 
classification task. 

I used tensorflow-probability to construct Bayesian dense layers that
would perform the classification of the pre-trained model's output. In
this project the Bayesian model captures the model and data uncertainty, 
but only in the features extracted by the pre-trained model. 

The pre-trained model I selected was the EfficientNetV2S. The 
EfficientNet models were originally developed in 
[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
as a solution to the problem of increasing the performance of neural networks
by increasing the width, depth, and resolution. This usually leads to the model
containing more parameters and decreasing efficiency without the performance 
gains. 

EfficientNets performed what as known as 'compound scaling', which makes the
network scaled up in depth, resolution, and width, without the performance costs
of a similarly performing but larger parameter model. The compound-scaling that
gives EfficientNet models their performance comes from stacking specific
layers into what the authors call 'modules'. These modules and the layers they are
built with are shown in the figure after this paragraph.

<figure>
    <img width="100%" src="https://miro.medium.com/v2/resize:fit:2000/format:webp/1*cwMpOJNhwOeosjwW-usYvA.png" 
alt='image of module types and their layers for efficientnet models'/>
</figure>

An EfficientNet model can be constructed by stacking these modules in different 
configurations called Blocks. The baseline EfficientNet-B0 has 7 stacked
configurations of blocks, with each subsequent model designation a note on how many 
modules have been stacked in each of the 7 block sections. 

The EfficientNetV2 networks were subsequently developed using the architectures
of the EfficientNet as a starting point in [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/pdf/2104.00298.pdf).
The creators of EfficientNetV2 jointly optimized network training speed and parameter
efficiency by combining training-aware neural architecture search and scaling. Progressive
learning, as introduced by the authors, is also a way to compensate for accuracy drops that
occur during speedups of training brought on by increasing image size.

I used the 'ImageNet' weights that exist for this model, not including the final layers
used for the final classification, as this is where I will append my Bayesian dense layers
to perform Bayesian classification on the extracted features of the images. The chest
X-ray images are imported using the Tensorflow-Keras *image_dataset_from_directory*
in order to save on local memory on my PC. I load the Pneumonia chest X-rays 
and reformat the dimensions to be 299x299 pixels, with 3 channels per image for the 
R,G,B colors, which are the same dimensions used by the researchers in the original
chest-xray paper.

I don't need to do a train-test split on this dataset since it comes pre-apportioned
with 5216 images in the training set (1341 Normal, 3875 Pneumonia) and 624 images in 
the test set (234 Normal, 390 Pneumonia). I make use of Keras data augmentation layers 
to augment the images used during training and insert them before the images have their
features extracted by the pre-trained neural network. The images have their contrast 
randomly modified by up to 30%. The second augmentation is translating the images by up
to 30% in both the x and y direction. The third augmentation randomly flips images
horizontally. The imbalanced number of normal to pneumonia+ images needs to be corrected to ensure that
the final classification isn't biased towards a diagnosis. The final class weights are 
1.95 for 'Normal' and 0.67 for 'Pneumonia'. 


**STILL UNDER CONSTRUCTION**{: .notice--success}







