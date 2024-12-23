---
title: 'Using Explainable AI Techniques To Understand Neural Network Learning'
date: 2024-11-01
permalink: /posts/2024/11/blog-post-1/
tags:
  - Explainable AI
  - Deep Learning
  - MRI
---

Abstract
======

In computational healthcare, machine learning algorithms have been developed for tumor grading, segmentation, and classification tasks leveraging Magnetic Resonance Imaging (MRI) data. In recent years, deep neural networks have been proposed for these tasks; however, these models lack transparency, making it difficult for clinicians to develop a mechanistic understanding of the models’ prediction logic. To address the need for transparency and interpretability in high-stakes clinical settings, we implement three local explainable AI methods in Tensorflow. Our main contribution is the application of Saliency Maps, Gradient-weighted Class Activation Mappings (Grad-CAM), and Counterfactual Explanations for helping neurologists interpret a Convolutional Neural Network (CNN) for brain tumor classification. We corroborate perspectives on the perceived utility of these explanations by drawing on our visualized explanations as evidence. In the process, we hope to provide algorithm developers with tools for explaining black-box computer vision algorithms while underscoring the limitations of current approaches in the context of medical image analysis problems.

Introduction
======

For more than half a century, computational healthcare has focused on using algorithms to emulate clinicians' expertise. Expert systems developed in the early 1970’s such as INTERNIST-1, CASNET, and MYCIN initiated efforts to use algorithms to assist clinicians in diagnosing medical conditions. For example, the INTERNIST-1 system used an extensive knowledge base covering roughly 75% of all known diseases in internal medicine to achieve this task. Each disease in the knowledge base had a disease profile submitted by a medical expert, which detailed manifestations (i.e. symptoms) of the disease and a rating of their correlation with the presence of the disease. Clinicians would enter a list of observed and unobserved manifestations for the system to formulate and rank disease “hypotheses”. 

A major limitation hindering the real-world deployment of INTERNIST-1 was the lack of explanations of its reasoning, as noted by Miller et. al. While the system provided a list of hypothesized diseases and the input manifestations that supported the hypotheses, clinicians had difficulty understanding how the system formulated hypotheses due to key information omitted from the knowledge base. For instance, the knowledge base did not distinguish between manifestations that predisposed individuals to a given disease versus manifestations that resulted from the disease. Thus, clinicians could not probe the system to determine causal links between observed manifestations and a hypothesized disease. Furthermore, the system did not consider time and anatomy, which clinicians rely heavily on for making and explaining diagnoses.

The reception to INTERNIST-1 highlights a desire for explainability that has long accompanied algorithms supporting high-stakes decisions - a desire being fulfilled by the research field of explainable AI (XAI). XAI research has focused on assessing and enabling human understanding of Artificial Intelligence (AI) systems.  While modern AI systems have evolved from 20th-century expert systems to machine learning models that solve complex problems by learning patterns from data, the goal of explainable AI remains the same. The two main goals of XAI as outlined by the Defense Advanced Research Projects Agency (DARPA), are to (1) develop accurate machine learning models that are explainable and (2) make explanations human-centric. Significant research efforts have been made to define explainability with quantitative metrics and critique definitions from a psychological perspective. The criteria that make an explanation human-centric are subjective and vary based on the application of explanations (purpose of explanations, explanation format, and audience)

Explainable AI has gained significant attention in the last decade with new regulations concerning the explainability of AI systems. Regulations such as the European Union’s General Data Protection Regulation (GDPR) have effectively conferred upon end users of machine learning systems the  “right to explanation”  (Goodman and Flaxman 2017). Supporting this regulation, draft standards such as ISO/TR 4804 (2020) propose explicit requirements for the explainability of machine learning components. While standards for AI explainability in healthcare have not yet been adopted, the Food and Drug Agency (FDA), which oversees the certification and approval of medical devices, has stated that an “appropriate level of transparency” for algorithms aimed at users is necessary.

The proliferation of deep neural networks (DNNs) has also contributed to the growth of explainable AI. DNNs have attained state-of-the-art performance on benchmark datasets in computer vision, such as the ImageNet Large Scale Visual Recognition Challenge (ILSVRC). These pivotal results have catalyzed the application of DNNs in high-stakes settings from medical image analysis to autonomous driving. Deep learning models have experienced a faster development rate than other ML methods such as Decision Trees and Linear Models, partly due to less time spent on feature engineering. Neural network models can learn representations of data in an unsupervised manner, making them effective in solving learning problems. However, these representations are often not readily interpretable by humans. In addition, the process for using these representations to obtain the model’s final output is confounded by multiple “hidden layers” encountered in deep neural network architectures. As such, researchers have used the term “black box” to describe the explainability of deep neural networks.

Despite being labeled as “black boxes”, deep neural networks have been applied successfully for tasks in medical image analysis. Medical image analysis aims to extract information from medical images using image processing techniques to assist clinical diagnoses. Common imaging modalities include X-ray, Computed Tomography (CT), and Magnetic Resonance Imaging (MRI). Some examples of deep neural network architectures that have achieved state-of-the-art performance include U-NET and CASS architectures, for brain tumor segmentation and classification tasks, respectively. Both tasks serve a complementary goal of helping neurologists diagnose brain tumors, with the latter task being the focus of this paper.

In the brain tumor classification problem, the goal is to classify a scan of a patient’s brain based on tumor phenotype. Model performance for this task is measured by classification error and F1 score. Convolutional Neural Networks (CNNs) are a popular choice of model for this task and play a crucial role in the CASS architecture for brain tumor classification. In this paper, a CNN architecture proposed by Badza and Batarovic is implemented and used for evaluations. This architecture contains only 3/100 of the parameters of CASS - simplifying the model development process - while still achieving an average accuracy of 96.56%  and an average F1 score of .961 on a dataset of over 3000 MRI slices. However, since CASS and the CNN model proposed by Badza and Batarovic were trained on different datasets, we refrain from comparing the performance of CASS with Badza and Batarovic’s model which we use in our experiments. 

Instead, we will evaluate the chosen CNN model’s explainability - a term frequently encountered in the XAI literature. Explainability means that “a model or evidence for a decision output is available and can be understood by end users”, as defined by Bruckert et. al. Here, understanding refers to mechanistic understanding (i.e. understanding how the model uses evidence to make predictions). Explanations of a model’s logic in the context of a specific prediction (e.g. “Why was this MRI slice classified as ‘glioma’?”) are called local explanations; explanations of a model’s logic in general are called global explanations. Models are said to be interpretable if they are explainable globally and locally, and if users can also develop a functional understanding of a model (i.e. by understanding why the model made a certain prediction). In the context of deep learning models, interpretability is a more difficult, potentially unfeasible condition to achieve. In contrast, many approaches have been developed to enhance the explainability of deep learning models.

The approaches to explaining deep learning models are predominantly local techniques, which can be grouped into feature attribution techniques, counterfactual explanations, and influential samples. Feature attribution techniques measure the importance of a feature to a model’s prediction - notable techniques include LIME, SHAP, saliency maps, and Grad-CAM. These techniques use different approaches to calculate feature importance. For instance, Grad-CAM measures feature importance through gradients, which are quantities obtained from making small perturbations to input features and measuring the change in the network’s output. Counterfactual explanations explain a model’s prediction for a given data point by finding the most similar data point that produces a different outcome under the model. Finally, techniques based on influential samples find data points most influential to a model’s prediction for a specific data point.

While previous works have assessed explainability techniques for deep neural networks, these works typically evaluate explanations on datasets such as the CIFAR-10 dataset of everyday objects, with no clear purpose for explanations in mind. Little research has assessed explanations for medical image analysis models and evaluated the explanation's ability to reconcile clinicians’ disagreement with clinical decision support systems. Thus, we aim to enhance the body of work evaluating explainable AI techniques for improving clinicians’ confidence in black box models deployed for clinical decision support. To achieve this goal, we make the following contributions:

1. We implement a convolutional neural network (CNN) architecture in Tensorflow for classifying brain tumors from slices of T2 contrast-weighted MRI scans.
2. We implement Saliency Maps, Grad-CAM, and LIME to explain the classifier’s prediction logic.


The outline of the paper is as follows. We begin by describing the dataset of MRI slices, the chosen model architecture, and the theory behind the three selected explainability techniques in the “Methods” section. Next, we will present sample visualizations of explanations obtained for a subset of data points from the dataset and interpret the explanations in the “Results” section. To assess the utility of the explanations, we underscore the limitations of the chosen methods remark on the limitations of the chosen methods, and use our obtained explanations to underscore these limitations. Finally, we suggest areas for future research.


<!-- 
References
======

[1]Brain Tumour Registry of Canada. https://braintumourregistry.ca/, 2019. Accessed: 2023-10-01. 

[2] E. S. Biratu, F. Schwenker, Y. M. Ayano, and T. G. Debelee, “A survey of brain tumor seg-
mentation and classification algorithms,” J Imaging, vol. 7, Sept. 2021.

[3] Milica M Badža and Marko Č Barjaktarović.
Classification of brain tumors from mri images using a convolutional neural network.
Applied Sciences, 10(6):1999, 2020.

[4] Yusuf Brima and Marcellin Atemkeng.
Visual interpretable and explainable deep learning models for brain tumor mri and covid-19 chest x-ray images.
2023. -->


Presentation Poster
======

![final poster](/files/final_poster.png)
