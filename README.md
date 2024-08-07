## Brain Tumor Classifiaction Using Machine Learning and Deep Learning 🧠

This a group project created as a part of **Applied Machine Learning for Data Scientists** course at **University of North Texas** by:           
[Chandrahaas](https://www.linkedin.com/in/chandrahaas-kalanadhabhatla/),
 [Pranav](https://www.linkedin.com/in/pranav-moses-2142b7154/)
 and [Sabeeha](https://www.linkedin.com/in/sabiha-tabassum-shaik-23a105241/)
 

---

**Data Gathering**:

The data set used for this was taken from: (https://universe.roboflow.com/work-tqclg/tumor-cjxoh/dataset/1) , which had more than 10,000 images for training, testing and validation.

**Pre-Processing**:

In this project we are using two approaches: 1. Traditional Machine learning approach 2. Deep Neural Network approach. Data pre processing the step where all the data in the dataset is converted to be consistent with each other and in a format that can be used to train the model without any issues. Data preprocessing techniques invlove reading an image, by default image data is not in a readable format for the machine learning models, we make
use of open source libraries such as matplotlib or openCV,
we have used openCV to read images in this project, it
converts the image into a matrix of numbers representing
each pixel in the image. The next step after reading the
image is to resize them, we want all the images in the dataset
to have the same resolution so we resize them to a consistent
number like 150X150 or 128X128. By the end of this step we
will end up having an object with 150X150X3 dimension
matrix. Then as the last step we normalize the pixel values
in the image, this helps the model better differentiate bright
pixels from the light ones.


**Feature Engineering**:

To enhance model performance in MRI image analysis, we applied feature engineering techniques such as Gabor and Sobel filters to accentuate specific textures and edges, enriching the feature set for machine learning models. Gabor filters, with varying orientations and frequencies, highlighted textural patterns, while Sobel filters detected edges by identifying sharp brightness contrasts. These pre-processing steps directed machine learning models to the most relevant image aspects. Deep neural networks, with their multi-layered architecture, automatically learned and identified significant patterns within the raw MRI data. We used EfficientNetB0, a pre-trained neural network model for image classification, along with Support Vector Machine, Random Forest, and Naive Bayes Multinomial classifiers, adapting them for our project's inputs and outputs.

**Modeling**:

For machine learning models, we
made use of sklearn to import the models and train them,
but for deep neural network, we had to import a pre-trained
model and add layer to it to make it suitable for our project’s
inputs and outputs. In this project, we have use a total
of four model: EfficientNetB0 model, which is a pretrained
neural network based model which is extensively
used for image classifications, Support Vector Machine classifier,
Random Forest classifier and Naive Bayes Multinomial
classifier.

**Evaluation**:

After training the model, we assessed its proficiency using test data by generating predictions for unseen examples and comparing them against the ground truth labels. This evaluation involved calculating performance metrics such as accuracy, precision, and recall, each providing insights into the model's strengths and weaknesses. To gain a comprehensive understanding of the model's performance across different classes, we used a confusion matrix, which summarizes the distribution of correct (True Positives and True Negatives) and incorrect (False Positives and False Negatives) predictions. This allowed us to calculate the metrics necessary to evaluate the classification model's performance.

**Results**:

![image](https://github.com/user-attachments/assets/32017a53-61d8-4cb6-a2ba-6f77a4a1043f)
![image](https://github.com/user-attachments/assets/93634cf3-b38f-4a9f-96cb-4dd703dd6afe)
![image](https://github.com/user-attachments/assets/f50d8c84-964c-4692-a7a3-512031684efe)

![image](https://github.com/user-attachments/assets/56a6e9b3-3182-4050-882a-86ef8de9f8bc)


Our primary aim was to create a machine-learning model capable of accurately detecting brain tumors, designed to handle large volumes of MRI scans typically processed by neurospecialists. The first milestone was to address the issue of early detection failure by providing a reliable automated method for brain tumor detection. Comparing the accuracy and F-1 scores of four models, we found that neural networks achieved the highest scores for both metrics. However, neural networks require GPU access, whereas SVM provides the best results when using a CPU.

