---
title: "Probabilistic Detection of Pneumonia in Chest X-Rays"
excerpt: "Sometimes accuracy isn't everything. <br/><img src='/images/pneumonia_cxr_nn/pneumonia_project_cover.png'>"
collection: portfolio
---
{%include base_path%}

Pneumonia is an infection wherein one or both of the lungs are incapacitated by fluids in the air-sacs of the lung. Pneumonia accounts for about 4 million deaths yearly across all age groups. Pneumonia is usually diagnosed by examination of chest X-rays wherein Pneumonia appears as cloudiness within the lungs.

<!-- ![sidebyside xrays or normal/pneumonia chests]('/pneumonia_cxr_nn/normal_pneumonia_cxr.png' "normal chest x-ray next to a chest x-ray exihiting pneumonia") -->
<center>
<figure>
    <img width="100%" src="/images/pneumonia_cxr_nn/normal_pneumonia_cxr.png" alt='normal chest x-ray sitiated next to a chest x-ray exhibiting pneumonia'/>
</figure>
</center>

Automated medical image diagnoses streamlines the usage of human medical resources. In this case a radiologist who visually inspects the X-rays to see if there is the presence of pneumonia. In the manual case this radiologist would give equal attention to each image they receive, and issuing a diagnosis or lack thereof. It's more important for the radiologis to give their attention to the possible pneumonia x-rays. An AI system that could pre-classify chest X-rays could expedite the diagnostic workflow of the radiologist and reduce times between examination, diagnosis, and most importantly, treatment. 

However a neural network classifier that outputs just a 'positive' or 'negative' diagnosis is **not** gonna cut it. It's important to understand here that non-Bayesian neural networks output classifications which do not take into account either the model uncertainty (otherwise called  _epistemic uncertainty_), or the uncertainty in the image data itself (called  _aleatoric uncertainty_).

A Bayesian neural network (_hereafter_  BNN) models its weights and biases as distributions and not as single values. A BNN can appropriately define the shape of these distributions which propogate into its classifications. The BNN's quantifiable uncertainty on its classifications provides certain benefits such as enhanced confidence in predictions, adaptability to a broad array of data of differing quality, and clearer insight into how the BNN functions with any False Positives or False Negatives, to name a few.

I used the Tensorflow-Probability python library to construct my Bayesian convolutional neural network (_hereafter_  BCNN) with probabilistic layers in the Tensorflow sequential model framework. The BCNN for this project has a similar structure to the BCNN I built for the Intel landscape image classification <a href="https://kjaehnig.github.io/portfolio/projectC_bayesian_cnn" style="color: #458900">'portfolio project'</a>. That project focused on delivering comparable performance on correctly classifying 6 different types of images. This project will _slightly_ deviate from that. 

I choose to orient this project on maximizing the detection rate of chest x-rays that exhibit Pneumonia and minimizing the occurance of False Negatives, that is, where a chest x-ray which has Pneumonia is classified as 'Normal' (this is also called a Type II error). I base this decision on the discrepancy between the real-world 'cost' of a False Negative and a False Positive when it comes to disease/condition detection. Untreated Pneumonia can be deadly, especially amongst the young, the old, and the immunocompromised. So ensuring that as many of the positive cases get the **correct** diagnosis is the goal.

This means that the primary metric of performance for this BCNN will not be accuracy, which is typically written as

$$
Accuracy = {TP + TN \over TP + TN + FP + FN}
$$

where TP, TN, FP, and FN stand for True Positive, True Negative, False Positive, and False Negative, respectively. Instead we want to focus the metrics, _Precision_ and _Recall_, which are defined as follows:

$$
Precision = {TP \over TP + FP},         Recall = {TP \over TP + FN}
$$

**STILL UNDER CONSTRUCTION**{: .notice--success}







