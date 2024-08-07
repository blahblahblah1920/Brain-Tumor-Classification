## Brain Tumor Classifiaction Using Machine Learning and Deep Learning 📈

This a group project created as a part of **Applied Machine Learning for Data Scientists** course at **University of North Texas** by:           
[Chandrahaas](https://www.linkedin.com/in/chandrahaas-kalanadhabhatla/),
 [Pranav](https://www.linkedin.com/in/pranav-moses-2142b7154/)
 and [Sabeeha](https://www.linkedin.com/in/sabiha-tabassum-shaik-23a105241/)
 

---

**DATA GATHERING**:

The data set used for this was taken from: (https://universe.roboflow.com/work-tqclg/tumor-cjxoh/dataset/1) , which had more than 10,000 images for training, testing and validation.



**PRE-PROCESSING**:

In this project we are using two approaches: 1. Traditional Machine learning approach 2. Deep Neural Network approach. Data pre processing the step where all the data in the dataset is converted to be consistent with each other and in a format that can be used to train the model without any issues. Data preprocessing techniques invlove reading an image, by default image data is not in a readable format for the machine learning models, we make
use of open source libraries such as matplotlib or openCV,
we have used openCV to read images in this project, it
converts the image into a matrix of numbers representing
each pixel in the image. The next step after reading the
image is to resize them, we want all the images in the dataset
to have the same resolution so we resize them to a consistent
number like 150X150 or 128X128. By the end of this step we


**MODELLING**:

For modelling, we used the following supervised machine learning models:
1.	Linear Regression (RMSE: 1397.29, R2: -400409.34)
2.	Polynomial Regression (RMSE: 4609286.85, R2: -4357111416313.00)
3.	Random Forest Regressor (RMSE: 0.93, R2: 0.82)
4.	Support Vector Machine Regressor (RMSE: 1.59, R2: 0.49)
5.	K-Nearest Neighbour Regressor (RMSE:1.15, R2: 0.73)



**EVALUATION**:

We can see each models performance represented with RMSE and R2 score.
From the above observation we concluded that the linear models displayed their unsuitability for this dataset due to the non-linear nature of the data. The high value for R2 for the RF, SVM and KNN indicates the strong explanatory power of the model. Both RMSE and R2 scores looked promising for the non-linear models.


