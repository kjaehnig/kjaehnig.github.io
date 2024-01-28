---
title: "Probabilistic Detection of Pneumonia in Chest X-Rays"
excerpt: "Sometimes accuracy isn't everything. <br/><img src='/images/pneumonia_cxr_nn/pneumonia_project_cover.png'>"
collection: portfolio
---

Pneumonia is an infection wherein one or both of the lungs are incapacitated by fluids in the air-sacs of the lung. Pneumonia accounts for about 4 million deaths yearly across all age groups. Pneumonia is usually diagnosed by examination of chest X-rays wherein Pneumonia appears as cloudiness within the lungs.

|:-:|
|![sidebyside xrays or normal/pneumonia chests]('/images/pneumonia_cxr_nn/normal_pneumonia_cxr_sidebyside.png' "normal chest x-ray next to a chest x-ray exihiting pneumonia")|

Automated medical image diagnoses streamlines the usage of human medical resources. In this case a radiologist who visually inspects the X-rays to see if there is the presence of pneumonia. In the manual case this radiologist would give equal attention to each image they receive, and issuing a diagnosis or lack thereof. It's more important for the radiologis to give their attention to the possible pneumonia x-rays. An AI system that could pre-classify chest X-rays could expedite the diagnostic workflow of the radiologist and reduce times between examination, diagnosis, and most importantly, treatment. 

However a neural network classifier that outputs just a 'positive' or 'negative' diagnosis is **not** gonna cut it. It's important to understand here that non-Bayesian neural networks output classifications which do not take into account either the model uncertainty (otherwise called _epistemic uncertainty_), or the uncertainty in the image data itself (called _aleatoric uncertainty_).