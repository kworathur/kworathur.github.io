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

For more than half a century, computational healthcare has focused on using algorithms to emulate clinicians' expertise. Expert systems developed in the early 1970’s such as INTERNIST-1 and MYCIN initiated efforts to use algorithms to assist clinicians in diagnosing medical conditions [1]. For example, the INTERNIST-1 system proposed by Miller et. al used an extensive knowledge base covering roughly 75% of all known diseases in internal medicine to achieve this task [2]. Each disease in the knowledge base had a disease profile submitted by a medical expert, which detailed manifestations (i.e. symptoms) of the disease and a rating of their correlation with the presence of the disease [2]. Clinicians would enter a list of observed and unobserved manifestations for the system to formulate and rank disease “hypotheses” [2]. 

A major limitation hindering the real-world deployment of INTERNIST-1 was the lack of explanations of its reasoning, as noted by its developers, Miller et. al [3]. While the system provided a list of hypothesized diseases and the input manifestations that supported the hypotheses, clinicians had difficulty understanding how the system formulated hypotheses due to key information omitted from the knowledge base. For instance, the knowledge base did not distinguish between manifestations that predisposed individuals to a given disease versus manifestations that resulted from the disease [2]. Thus, clinicians could not probe the system to determine causal links between observed manifestations and a hypothesized disease. Furthermore, the system did not consider time and anatomy, which clinicians rely heavily on for making and explaining diagnoses [2].

The reception to INTERNIST-1 highlights a desire for explainability of AI algorithms used in high-stakes contexts, which is being addressed by the research field of explainable AI (XAI). XAI research has focused on assessing and enabling human understanding of Artificial Intelligence (AI) systems. While modern AI systems have evolved from 20th-century expert systems to machine learning models, the goal of explainable AI remains the same. The two main goals of XAI, as outlined by the Defense Advanced Research Projects Agency (DARPA), are to (1) develop accurate machine learning models that are explainable and (2) make explanations human-centric [4]. XAI research leverages ideas from statistical learning, cognitive science, philosophy, and psychology to achieve these goals [5]. This research is crucial in promoting the safe deployment of machine learning systems for real-world use by making it easier to debug and detect biases in these systems.

Indeed, explainability forms a pillar of trustworthy machine learning systems and thus is referenced frequently in AI regulation. Recent regulations such as the European Union’s General Data Protection Regulation (GDPR) have been proposed to protect the subjects of algorithmic decisions, effectively conferring upon them the  “right to explanation” [6]. Supporting this regulation, draft standards such as ISO/TR 4804 (2020) propose explicit requirements for the explainability of machine learning components [7]. While standards for AI explainability in healthcare have not yet been adopted, the Food and Drug Agency (FDA), which oversees the certification and approval of medical devices in the United States, has stated that an “appropriate level of transparency” for algorithms aimed at users is necessary [8].

Although the research field of explainable AI is well-established, the field has recently had to adapt to new challenges in explainability brought by the proliferation of deep neural networks (DNNs). DNNs have attained state-of-the-art performance on benchmark datasets in computer vision, such as the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) [9]. These pivotal results have catalyzed the application of DNNs in high-stakes settings from medical image analysis to autonomous driving. Deep learning models have experienced a faster development rate than other ML methods such as Decision Trees and Linear Models, partly due to less time spent on feature engineering. Neural network models can learn representations of data in an unsupervised manner, making them effective in solving learning problems. However, these representations are often not readily interpretable by humans. In addition, the process for using these representations to obtain the model’s final output is confounded by multiple “hidden layers” encountered in deep neural network architectures. As such, researchers have used the term “black box” to describe the explainability of deep neural networks [10].

Despite being labeled as “black boxes”, deep neural networks have been applied for medical image analysis with promising results. Medical image analysis aims to extract information from medical images using image processing techniques to assist clinical diagnoses. Common imaging modalities include X-ray, Computed Tomography (CT), and Magnetic Resonance Imaging (MRI). Some examples of deep neural network architectures that have achieved state-of-the-art performance include U-NET [11] and cross-architectural Self-Supervision (CASS) architectures [12], for brain tumor segmentation and classification tasks, respectively. Both tasks complement each other in helping neurologists diagnose brain tumors, with the latter task being the focus of this paper.

In the brain tumor classification problem, the goal is to classify a scan of a patient’s brain based on tumor phenotype. Model performance for this task is measured by classification error and F1 score. Convolutional Neural Networks (CNNs) are a popular choice of model for this task and play a crucial role in the CASS architecture for brain tumor classification [12]. In this paper, a CNN architecture proposed by Badza and Batarovic is implemented and used for evaluations [13]. This architecture contains only 3/100 of the parameters of CASS - simplifying the model development process - while still achieving an average accuracy of 96.56%  and an average F1 score of 0.961 on a dataset of over 3,000 MRI slices [13]. However, since CASS and the CNN model proposed by Badza and Batarovic were trained on different datasets, we refrain from comparing the performance of CASS with Badza and Batarovic’s model. 

Instead, we will evaluate the chosen CNN model’s explainability - a term frequently encountered in the XAI literature [5]. Explainability means that “a model or evidence for a decision output is available and can be understood by end users”, as defined by Bruckert et. al [14]. Here, “understanding” refers to mechanistic understanding (i.e. an understanding of how the model uses evidence to make predictions). Explanations can have varying locality -  explanations of a model’s logic in the context of a specific prediction (e.g. “Why was this MRI slice classified as ‘glioma’?”) are called local explanations and explanations of a model’s logic in general are called global explanations [5]. Models are interpretable if they are explainable globally and locally, and users can also develop a functional understanding of a model (i.e. by understanding why the model made a certain prediction) [5]. The “black box” nature of deep learning models often precludes interpretability; however, many approaches have been developed to enhance these models’ explainability.

Current approaches to explaining deep learning models are predominantly local techniques, which can be grouped into feature attribution techniques, counterfactual explanations, and influential samples. We will focus on feature attribution and counterfactual techniques for producing explanations. Feature attribution techniques measure the importance of a feature to a model’s prediction - notable techniques include saliency maps [15], Layer-Wise Relevance Propagation (LRP) [16], and Gradient-Weighted Class Activation Mapping (Grad-CAM) [17]. These techniques use different approaches to calculate feature importance. For instance, Grad-CAM measures feature importance through gradients, which are quantities obtained from making small perturbations to input features and measuring the change in the network’s output. Counterfactual explanations explain a model’s prediction for a given data point by finding the most similar data point that produces a different outcome under the model. 

In this paper, we aim to assess the ability of explanations to improve clinician’s trust in black-box clinical decision-support tools. While previous works have assessed explainability techniques for black box models, evaluation of explanations is limited mostly to benchmark datasets with no clear purpose for explanations in mind. We are interested specifically in assessing explanations’ ability to reconcile clinicians’ disagreement with neural network models for clinical diagnosis. To achieve this goal, we make two key contributions:

1. We implement a convolutional neural network (CNN) architecture in Tensorflow for classifying brain tumors from slices of T2 contrast-weighted MRI scans.
2. We implement two permutation-based feature importance techniques (Saliency Maps and Grad-CAM), and Counterfactual Explanations to explain the classifier’s prediction logic.
   
The rest of the paper is organized as follows:
1. In the “Methods” section, we describe the dataset of MRI slices, the chosen model architecture, and the theory behind the three selected explainability techniques. 
2. Next in the “Results” section we will present sample visualizations of explanations obtained for a subset of data points from the dataset. 
3. In the “Discussion” section, we interpret the explanations, underscore their limitations, and propose desiderata for explanations of medical image models.


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
