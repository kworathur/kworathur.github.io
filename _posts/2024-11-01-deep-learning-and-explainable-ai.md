---
title: 'Using Explainable AI Techniques To Understand Neural Network Learning'
date: 2024-11-01
permalink: /posts/2024/11/blog-post-1/
tags:
  - Explainable AI
  - Deep Learning
  - MRI
---

# Abstract

In computational healthcare, machine learning algorithms have been developed for tumor grading, segmentation, and classification tasks leveraging Magnetic Resonance Imaging (MRI) data. In recent years, deep neural networks have been proposed for these tasks; however, these models lack transparency, making it difficult for clinicians to develop an understanding of these models’ prediction logic. To address the need for transparency and interpretability in high-stakes clinical settings, we implement two local explainable AI methods in Tensorflow. Our main contribution is the application of saliency maps and Gradient-weighted Class Activation Mappings (Grad-CAM) for helping neurologists interpret a Convolutional Neural Network (CNN) for brain tumor classification. Using our explanations, we find that our proposed model is influenced by MR imaging artifacts and other features that do not correlate well with tumor type. We emphasize the value of explainable AI tools in identifying sources of bias in training data, while underscoring the limited insight feature-attribution techniques provide clinicians.

# Introduction

For more than half a century, computational healthcare has focused on using algorithms to emulate clinicians' expertise. Expert systems developed in the early 1970’s such as MYCIN [1] and INTERNIST-1 [2] initiated efforts to use algorithms to assist clinicians in diagnosing medical conditions. For example, the INTERNIST-1 system proposed by Miller et. al used an extensive knowledge base covering roughly 75% of all known diseases in internal medicine to achieve this task [2]. Each disease in the knowledge base had a disease profile submitted by a medical expert, which detailed manifestations (i.e. symptoms) of the disease and a rating of their correlation with the presence of the disease [2]. Clinicians would enter a list of observed and unobserved manifestations for the system to formulate and rank disease “hypotheses” [2]. 

A major limitation hindering the real-world deployment of INTERNIST-1 was the lack of explanations of its reasoning, as noted by its creators [3]. While the system provided a list of hypothesized diseases and the input manifestations that supported the hypotheses, clinicians had difficulty understanding how the system formulated hypotheses due to key information omitted from the knowledge base. For instance, the knowledge base did not distinguish between manifestations that predisposed individuals to a given disease versus manifestations that resulted from the disease [2]. Thus, clinicians could not probe the system to determine causal links between observed manifestations and a hypothesized disease. Furthermore, the system did not consider time and anatomy, which clinicians rely heavily on for making and explaining diagnoses [2].

The reception to INTERNIST-1 highlights a desire for explainability of AI algorithms used in high-stakes contexts, which is being addressed by the research field of explainable AI (XAI). XAI research has focused on assessing and enabling human understanding of Artificial Intelligence (AI) systems. While modern AI systems have evolved from 20th-century expert systems to machine learning models, the goal of explainable AI remains the same. The two main goals of XAI, as outlined by the Defense Advanced Research Projects Agency (DARPA), are to (1) develop accurate machine learning models that are explainable and (2) make explanations human-centric [4]. XAI research leverages ideas from statistical learning, cognitive science, philosophy, and psychology to achieve these goals [5]. This research is crucial in promoting the safe deployment of machine learning systems for real-world use by making debugging and detecting biases in these systems easier.

Indeed, explainability forms a pillar of trustworthy machine learning systems and has been codified by regulations over the past decade. For instance, the European Union’s 2017 General Data Protection Regulation (GDPR) has been proposed to protect the subjects of algorithmic decisions, effectively conferring upon them the  “right to explanation” [6]. Supporting this regulation, draft standards such as ISO/TR 4804 (2020) propose explicit requirements for the explainability of machine learning components [7]. While standards for AI explainability in healthcare have not yet been adopted, the Food and Drug Agency (FDA), which oversees the certification and approval of medical devices in the United States, has stated that an “appropriate level of transparency” for algorithms aimed at users is necessary [8].

Although the research field of explainable AI is well-established, the field has recently had to adapt to new challenges in explainability brought by the proliferation of deep neural networks (DNNs). DNNs have attained state-of-the-art performance on benchmark datasets in computer vision, such as the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) [9]. These pivotal results have catalyzed the application of DNNs in high-stakes settings from medical image analysis to autonomous driving. Deep learning models have experienced a faster development rate than other ML methods such as Decision Trees and Linear Models, partly due to their unsupervised representation learning that reduces time spent on feature engineering. While effective for various learning problems, the intermediate representations learned by neural networks are not always amenable to human interpretation. In addition, the steps for using these representations to obtain the model’s final output are obfuscated by multiple “hidden layers” encountered in deep neural network architectures. As such, researchers have used the term “black box” to describe the explainability of deep neural networks [10].

Despite being labeled as “black boxes”, deep neural networks have been applied for medical image analysis with promising results. Medical image analysis aims to extract information from medical images using image processing techniques to assist clinical diagnoses. Common imaging modalities include X-ray, Computed Tomography (CT), and Magnetic Resonance Imaging (MRI). Some examples of deep neural network architectures that have achieved state-of-the-art performance include U-NET [11] and cross-architectural Self-Supervision (CASS) architectures [12], for brain tumor segmentation and classification tasks, respectively. Both tasks complement each other in helping neurologists diagnose brain tumors, with the latter task being the focus of this paper.


In the brain tumor classification problem, the goal is to classify a 2D MRI slice of a patient’s brain based on tumor phenotype. Model performance for this task is measured by classification error and F1 score. Convolutional Neural Networks (CNNs) are a popular choice of model for this task, having been the backbone for successful architectures for brain tumor classification such as CASS [12].  In this paper, a simpler CNN architecture proposed by Badza and Batarovic [13] is used for evaluations. This architecture contains only 3/100 of the parameters of CASS - simplifying the model development process - while still achieving an average accuracy of 96.56%  and an average F1 score of 0.961 on a dataset of over 3,000 MRI slices [13]. However, since CASS and the CNN model proposed by Badza and Batarovic were tested on different datasets, we refrain from comparing the performance of CASS with Badza and Batarovic’s model. 

Instead, this paper will evaluate the chosen CNN model’s explainability - a term frequently encountered in the XAI literature [5]. Explainability means that “a model or evidence for a decision output is available and can be understood by end users”, as defined by Bruckert et. al [14]. Here, “understanding” refers to mechanistic understanding (i.e. an understanding of how the model uses evidence to make predictions). Explanations of a model’s logic in the context of a specific prediction (e.g. “Why was this MRI slice classified as ‘glioma’?”) are called local explanations and explanations of a model’s logic in general are called global explanations [5]. Models are interpretable if they are explainable globally and locally, and users can also develop a functional understanding of a model (i.e. by understanding why the model made a certain prediction) [5]. The “black box” nature of deep learning models often precludes interpretability; however, many approaches have been developed to enhance these models’ explainability.

This paper will focus on local techniques for explaining deep learning models, particularly feature attribution techniques. Feature attribution techniques measure the importance of a feature to a model’s prediction - notable techniques include saliency maps [15], Layer-Wise Relevance Propagation (LRP) [16], and Gradient-Weighted Class Activation Mapping (Grad-CAM) [17]. These techniques use different approaches to calculate feature importance. For instance, Grad-CAM measures feature importance through gradients, which are quantities obtained from making small perturbations to input features and measuring the change in the network’s output. 

Numerous works have developed techniques for making clinical decision support systems more transparent to the benefit of clinicians. For instance, Xie et. Al designed and implemented XAI for Chest X-Rays, which aimed to provide computer-generated explanations for referring physicians similar to explanations provided by radiologists. In a survey with 39 referring physicians, the authors found that image-based comparisons (such as contrastive explanations, comparisons over time, comparisons between patients) received the highest rating from physicians in terms of utility. In contrast, explanations that highlighted regions of a chest X-ray without providing additional context were perceived as less helpful. Corroborating this perspective, Mertes et. Al found that compared to techniques such as counterfactual explanations, feature-attribution explanations provide limited insight: feature attribution techniques can explain what parts of a scan led to a model’s prediction but not why each highlighted part of the scan justifies the prediction.

Through our work, we acknowledge concerns about the utility of feature-attribution techniques highlighted by previous research, while showing how these types of explanations can provide valuable insight into data quality issues (e.g. bias, noise) which developers can act upon. To achieve this goal, we make three key contributions:

1. We implement a convolutional neural network (CNN) architecture in Tensorflow for classifying brain tumors from slices of T2 contrast-weighted MRI scans.
2. We implement two permutation-based feature importance techniques (Saliency Maps and Grad-CAM), and Counterfactual Explanations to explain the classifier’s prediction logic.
3. We provide a detailed analysis into data quality issues (e.g. imaging artifacts) highlighted by our explanations and propose possible improvements to our data-preprocessing pipeline to mitigate these issues.


The rest of the paper is organized as follows. In the “Methods” section, we describe the dataset of MRI slices, the chosen model architecture, and the theory behind the three selected explainability techniques. Next in the “Results” section we will present sample visualizations of explanations obtained for a subset of data points from the dataset. In the “Discussion” section, we interpret the explanations, corroborate their limitations as identified by previous works, and highlight their value in improving data quality.


# Methods


## Dataset

We begin by introducing the Magnetic Resonance Imaging (MRI) modality. MRI produces high-resolution imagery of the body’s soft tissues in a non-invasive manner. This imaging process repetitively stabilizes and disrupts the orientation of protons of water molecules in tissues using a uniform electromagnetic field and radio frequency (RF) pulses. The RF pulses excite the protons, which release energy measured as a signal which is then converted into signal intensity using the Fourier transform operator. When applied to the brain, MRI can detect the brain’s anatomical structures such as dense neuron structures, cerebrospinal fluid (CSF), and the axons connecting parts of the brain. MRI slices are obtained by measuring signals in a cross-section of the brain that slides to produce multiple slices for each scan.

The most common types of MRI scans are T1 and T2 weighted scans, which arise from different configurations of imaging parameters. The repetition time (TR) and time to echo (TE) determine whether a scan is T1-weighted or T2-weighted. Using a short TR and TE allows MRI to characterize tissues based on the time for protons to return to equilibrium after an RF pulse. T1 is defined as the rate at which protons return to equilibrium, thus tissues whose protons return to equilibrium faster than other tissues appear lighter in the scan. For example, dense tissue such as fat is highlighted in T1 scans while the brain’s fluid regions such as the CSF are dark. 

Next, we describe the brain tumor dataset used for model training. The dataset consists of 3,064 T1-weighted MRI slices obtained from Nanfang Hospital, Guangzhou, China, and General Hospital, Tianjin Medical University obtained between 2005 and 2010 . Slices of MRI scans were obtained by taking a cross-section of a three-dimensional scan in one of three planes: sagittal, axial, or coronal. Each slice in the dataset is labeled with one of three types of tumors: meningioma, glioma, and pituitary tumor. Figure 1 shows the balance of class labels in the training dataset; glioma tumors appear almost 2 times more often than pituitary tumors, the next most common label in the dataset. This evident class imbalance has implications on the model’s performance and error by tumor type. We discuss this imbalance in detail in the results section.

Finally, we describe the steps for pre-processing the training data before it is provided as input to the CNN model. First, the grayscale MRI slices are resized from 512 by 512 pixels to 256 by 256 pixels. Each slice is converted to a 3D tensor, where the first two dimensions of the tensor represent the width and height of the slice and the last dimension represents the single color channel used for grayscale pixel values. Next, the dataset size is tripled by adding two augmented versions of each MRI slice: one version is obtained from rotating the MRI slice 90 degrees counterclockwise and a second version is obtained by flipping the MRI slice over the x-axis. Last, each pixel value in the MRI slice is normalized by subtracting the mean of pixel values in the same position across all slices in the training dataset, and dividing by the standard deviation of these pixel values. 

## Model

The model used for classifying brain tumors is a convolutional neural network (CNN), which consists of four convolutional blocks stacked in succession. Figure 1 below visualizes the first two classification blocks of the CNN model. 

![Model Architecture Diagram](/files/arch.png)

The final output of the model is a probability distribution over three target classes: meningioma, glioma, and pituitary.

## Training and Validation Approach

* 20% of data was randomly selected to be held out as a test set. The remaining 80% was used for training and validation.

* To prevent leakage of test data into the training and validation sets, a second dataset splitting approach was taken, known as subject-wise cross validation. 

## Description of Explainable AI Methods

### Saliency Maps

Saliency maps applied to image data can provide a visual explanation of a model’s prediction by highlighting the most “salient” pixels of the input image. The saliency of a pixel is measured by the partial derivative of the highest class probability with respect to the pixel, which is given by equation (1)

$$S_{i, j} = \frac{\partial \max _{c \in C}(y^{c}))}{\partial X_{i, j}}$$

Equation (1) measures the change in the highest predicted class probability for an infinitesimal perturbation of an input pixel. Therefore, high saliency pixels produce a larger positive/negative change in the predicted class probability when perturbed compared to low saliency pixels. 

### Grad-CAM

Grad-CAM can also be used to produce visual explanations, and uses a two-pass process for measuring feature importance.

![Grad-CAM Explanation](/files/gradcam_explanation.png)

Grad-CAM uses a two-pass process for generating heatmaps as detailed in Figure 1. The first stage of heatmap generation (steps i) through iii))  measures the contribution of input features to predicted tumor class. In step i), the 2D MRI slice is pre-processed by resizing the image to 256 x 256 pixels and normalizing pixel values. In step ii), the MRI slice is processed by the downstream network layers before passing through the network’s final convolutional layer. The outputs of the final convolutional layer are passed through a non-linear activation function (e.g. Rectified Linear Units) to obtain 3D feature maps. Feature maps capture coarse-grained features learned by the network that are useful for determining tumor class. In step iii), the feature maps are passed through task-specific layers of the network to obtain a reactivity score. 

$$H_{\text{Grad-CAM}} = \text{ReLU}(\Sigma_{k} \sigma_{k} \cdot A_{k})$$

The second stage of heatmap generation (steps iv) through vi)) will coalesce the previously computed feature maps and subsequently attribute the  “pooled” feature map to regions of the input grid. The pooling of feature maps is performed via the weighted combination given by Equation 1. The weights k are determined by gradients, mathematical quantities that measure how much a small change in an input feature, such as a single atom’s charge, influences the reactivity score. In particular, k is the gradient of the predicted class probability, computed in step iii), with respect to feature map k, computed in step ii). Finally, the pooled feature map is upsampled using bilinear interpolation to have the same dimensions of the input MRI slice. 

# Results

## Training Results

Table 1 below summarizes the results of training with both the ‘one test’ and subject-wise cross validation dataset splitting approaches.

| Training Configuration     | Test Precision | Test Recall | Test Accuracy | Test F1 Score|
| ----------- | ----------- |  ----------- | ----------- |  ----------- | 
| One Test      | 90.34%       | 88.45% | 88.89% | 89.38%|

Table 1: AUROC metrics computed during training and inference show that the model is able to accurately classify tumors in an unseen set of patient scans.

## Heatmap Explanations

![Axial Glioma Heatmaps](/files/glioma_axial.png)

Figure 1: Saliency Map and Grad-CAM explanations for three scans of Glioma tumors, for which the model correctly predicted as a glioma tumor. The left column of MRI slices contains the tumor highlighted in red. The middle column contains the saliency map feature sensitivities, where features are highlighted according to the classifier’s sensitivity to the features. The right column shows the Grad-CAM “heatmaps”, in which regions colored in warmer colors (e.g. orange, red, dark red)  are more influential to the classification compared to regions colored in cooler colors (cyan, green, yellow).


Figure 1 highlights how model explanations can be used to assess whether models are using robust features for classification. Although the CNN model correctly classified each of the three glioma scans above, the saliency maps for each classification show that the model is sensitive to features that do not correlate with tumor type. For instance, the saliency map in the first row of Figure 1 has a large cluster of white pixels at the bottom left corner of the MRI slice. This saliency map communicates to a ML developer or neurologist that the probability of the scan containing a glioma tumor, as determined by the model, will increase or decrease significantly for a small change in the intensity of pixels at the bottom left corner of the MRI slice. Similarly, the saliency maps on the second third rows support the observation that the CNN model uses image backgrounds for classifying tumors. 

The Grad-CAM heatmaps in the right column of Figure 1 also highlight the model’s dependence on the image background for making classifications. The heatmaps show that when the predicted class is ‘glioma’, the regions of the scan with the strongest activations are those which are situated on the periphery of the brain and the edges of the MRI slices. One exception to this pattern is the Grad-CAM heatmap on the second row of Figure 1, in which the region with the strongest activation was located in close proximity to the tumor region. Despite this, the right edge of the scan shows large activations, further demonstrating the model’s sensitivity to image background for classification.


# Discussion

In this section, we discuss the value of the visualizations obtained in the ‘Results’ section for clinicians and developers. The visual nature of the saliency maps and Grad-CAM explanations make it easy for clinicians to determine which regions of the scan were used in the classification, however visualizations produced by these methods do not explain why a region was influential to the model’s prediction. This lack of justification can present issues to clinicians aiming to develop an understanding of the classification algorithm. Furthermore, visualizations such as those presented in the results section can challenge or contradict a clinician’s prior understanding of  machine learning, as the highlighted regions of the visualizations appear to have no data that a machine learning model could learn from.

On the other hand, the visualizations in the ‘Results’ section can be useful to developers who are in the debugging stage of development. Saliency maps and Grad-CAM heatmaps highlight when the trained model uses features that are not causal/robust, which gives developers actionable insights for improving their data pre-processing pipelines. For example, upon finding that margins of the MRI slices are frequently highlighted in saliency maps, a developer can perform input data perturbation with gaussian noise as a measure to prevent the model from relying heavily on background pixels for classification. Another approach developers can take is to zero out all pixels which belong to the background during data pre-processing.

# Conclusion

This paper implemented two local explainable AI methods, saliency maps and Grad-CAM, to explain a CNN model used for brain tumor classification. By analyzing the visualizations produced by our two chosen methods, we highlighted the benefits of model explanations for developers while acknowledging the limitations of feature-attribution based techniques such as Grad-CAM and saliency maps.


References
======

[1] Ayumu Asai, Masamitsu Konno, Masateru Taniguchi, Andrea Vecchione, and Hideshi Ishii. Computational healthcare: Present and future perspectives (review). Exp. Ther. Med., 22(6):1351, Dec. 2021. 

[2] D A Wolfram. An appraisal of INTERNIST-I. Artif. Intell. Med., 7(2):93–116, Apr. 1995.

[3] R A Miller, H E Pople, Jr, and J D Myers. Internist-1, an experimental computer-based diagnostic consultant for general internal medicine. N. Engl. J. Med., 307(8):468–476, Aug. 1982. 

[4] David Gunning and David W Aha. DARPA’s explainable artificial intelligence program. AI Mag., 40(2):44–58, June 2019.

[5] Gesina Schwalbe and Bettina Finzel. A comprehensive taxonomy for explainable artificial intelligence: A systematic survey of surveys on methods and concepts. May 2021. 

[6] Bryce Goodman and Seth Flaxman. European union regulations on algorithmic decision making and a “right to explanation”. AI Mag., 38(3):50–57, Sept. 2017.

[7] Road Vehicles Technical Committee ISO/TC 22. Iso/tc 22 road vehicles (2020) iso/tr 4804:2020: Road vehicles — safety and cybersecurity for automated driving systems — design, verification and validation. https://www.iso.org/standard/80363.html, 2020. 

[8] U.S. Food & Drug Administration. Proposed regulatory framework for modifications to artificial intelligence/machine learning (ai/ml)-based software as a medical device (samd). Technical report.
[9] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In F. Pereira, C.J. Burges, L. Bottou, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems, volume 25. Curran Associates, Inc., 2012.

[10] Cynthia Rudin. Stop explaining black box machine learning models for high-stakes decisions and use interpretable models instead. Nat. Mach. Intell., 1(5):206–215, May 2019. 

[11] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional networks for biomedical image segmentation. 2015. 

[12] Pranav Singh, Elena Sizikova, and Jacopo Cirrone. CASS: Cross-architectural Self-Supervision for medical image analysis. June 2022. 

[13] Milica M Badza and Marko C Barjaktarovic. Classification of brain tumors from MRI images  using a convolutional neural network. Appl. Sci. (Basel), 10(6):1999, Mar. 2020. 

[14] Sebastian Bruckert, Bettina Finzel, and Ute Schmid. The next generation of medical decision support: A roadmap toward transparent expert companions. Front. Artif. Intell., 3:507973, Sept. 2020. 

[15] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. 2013. 

[16] Sebastian Bach, Alexander Binder, Gr´egoire Montavon, Frederick Klauschen, Klaus-Robert M¨uller, and Wojciech Samek. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. PLoS One, 10(7):e0130140, July 2015. 

[17] Ramprasaath R. Selvaraju, Abhishek Das, Ramakrishna Vedantam, Michael Cogswell, Devi Parikh, and Dhruv Batra. Grad-cam: Why did you say that? visual explanations from deep networks via gradient-based localization. CoRR, abs/1610.02391, 2016.

Presentation Poster
======

![final poster](/files/final_poster.png)
