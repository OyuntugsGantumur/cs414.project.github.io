-   [Introduction](#introduction)
-   [Data Gathering](#data-gathering)
-   [Model Structure](#model-structure)
    -   [Base Model](#base-model)
    -   [Modifications](#modifications)
-   [Facial Recognition](#facial-recognition)
-   [Model Running](#model-running)
-   [Conclusion](#conclusion)
-   [Future Research](#future-research)
-   [Run Notebook in Google Colab](#run-notebook-in-google-colab)
-   [Youtube Video Link](#youtube-video-link)
-   [Inquiries](#inquiries)
-   [Data Sources](#data-sources)
-   [Acknowledgments](#acknowledgments)

# Introduction
(OYU)
- Introduction to our project, project goals and planning


# Data Gathering
To gather data and eventually train our model, we scoured the Kaggle website and found one [dataset](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data). The dataset consisted of around 35000 grayscale images sized 48x48 that were grouped into 7 categories of emotions: 0 - Angry, 1 - Disgust, 2 - Fear, 3 - Happy, 4 - Sad, 5 - Surprise, and 6 - Neutral. 

The dataset had been divided into 3 sets: Training, Public Test, Private Test. We used each of these sets for the training phase, the validation phase and finally the testing phase respectively.

<div class="datatable-begin"></div>

| Emotion | Set | Pixels |
| --- | --- | --- |
|2|Training|231 212 156 164 174 138 161 173 182 200 106 38...
|0|Training|70 80 82 72 58 58 60 63 54 58 60 48 89 115 121...
|4|Training|24 32 36 30 32 23 19 20 30 41 21 22 32 34 21 1...
|1|Public Test|23 53 89 184 173 173 146 194 123 170 97 45 74 34 75 44...
|4|Public Test|77 28 94 23 44 12 18 93 45 112 163 64 4 25 87 9...
|5|Public Test|4 12 0 0 0 0 0 0 0 11 26 75 3 15 23 28 48 50 58 84..
|6|Private Test|50 36 17 22 23 29 33 39 34 37 37 37 39 43 48 5...
|3|Private Test|30 28 28 29 31 30 42 68 79 81 77 67 67 71 63 6...
|0|Private Test|17 17 16 23 28 22 19 17 25 26 20 24 31 19 27 9...

<div class="datatable-end"></div>

We chose this dataset based on its relatively small size, its consistency and completeness. Since we were tasked to create a self-sustaining notebook and make the project run seamlessly without local downloads, we decided to upload all of the dataset files to our shared Github repository. As the extracted dataset csv file was too big to upload at once, we split the dataset into 15 chunks and pushed the files to our remote repository. You can see samples of the images of the dataset below:

<p align="center">
    <img src = "images/dataset_labelling.png" alt = "Dataset Images">
</p>

# Model Structure
(SHAKH)
## Base Model
To start off, we scoured the Internet in order to find an example code to start off with. Our main objective with this project is to optimize a given model in order to detect faces in other images containing human faces, so utilizing existing code was permissible. There were many solutions with varying accuracy results, and we decided on the code written by [Alpel Temel](https://www.kaggle.com/code/alpertemel/fer2013-with-keras) in Kaggle as it had a relatively high validation accuracy of 58% and used concepts we were relatively familiar with. 

The model itself is a sequential model that utilizes convolutional neural networks (CNNs), which is perfect for this project as CNNs are regarded to be very accurate at image recognition and classification. We have 2D convolution layers, where the kernel performs element wise multiplication and summation of results into a single output. For our activation function, we have a ReLU function in our hidden layers, as it circumvents the vanishing gradient problem, results in a better performance and converges at a faster rate than other activation functions. We have also discussed in class that the ReLU sometimes performs better than the hyperbolic tangent functions, and we have also found that to be the case in the process of hyperparameter tuning. The output layer is implemented with the softmax function and it suits our project as it is a multiclass classification problem - given the goal is to label each face with one of the seven emotions - and the softmax performs best when handling multiple classes. 

Our next goal was to find a way to reconfigure the model in order to increase the accuracy of the model. We added batch normalization in order to stabilize the inputs, which we believe helped the model’s output at the end. Also, we added max pooling to remove invariances, making the model detect the presence of features rather than its exact location. The specific change we made was to remove average pooling and only utilize max pooling because we believe the former wasn’t extracting important features and was causing the accuracy to be lower. Since we wanted to eventually test the model on images of tilted faces in different angles, we found max pooling to be crucial in the model’s accuracy. Moreover, we added additional layers and experimented to find the optimal solution, meanwhile considering the time we had. Utilizing the GPU engine present in Google Colab, we experimented with different number of layers and tracked the accuracies, which sometimes dropped off from the starting point of 58% accuracy. Eventually, we ended up with a model with over 20 million parameters. We ran the model and these were the results:

<p align="center">
    <img src = "images/model_result.png" alt = "Model Result">
</p>



# Facial Recognition
(OYU)
- Computer Vision library
- challenges
- our work-around


# Model Running
(OYU)
- showing model running image
- our usage of custom hand-picked images
- the results and accuracy of our model on those images


# Conclusion
(SHAKH)
- Some type of conclusion and future directions
- Ways to "improve" or change approach (handling side-face images in training)



# Run Notebook in Google Colab

Click the link below to run [our notebook](https://github.com/pard187/pard187.github.io/blob/master/Final_Project_Gormley_Giffin_Johnston_Saleh.ipynb) directly in Google Collab. No coding is required to run this notebook, you just need to run every code cell in order or simply click Runtime -> Run all and wait for all cells to run. 
Click the link below to run [our notebook](https://github.com/pard187/pard187.github.io/blob/master/Final_Project_Gormley_Giffin_Johnston_Saleh.ipynb) directly in Google Collab. No coding is required to run this notebook, you just need to run every code cell in order or simply click Runtime -> Run all and wait for all cells to run.

*Please note that since the US-Accident dataset we are using is too big, it was not convienient to download and import it to Google Drive or Github. Therefore, our notebook is pulling the data directly from Kaggle. A potential drawback to this method is that any changes to the dataset on Kaggle will affect the ability of the analysis in this notebook to be replicated. At the time of analysis, the US-Accidents dataset was last updated July 9, 2020. Were the dataset to be altered at a later date, then the conclusions drawn as a part of this analysis might change. We have however, stored a version of the US Accidents data in this GitHub repository for access at a later date.*

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/pard187/pard187.github.io/blob/master/Final_Project_Gormley_Giffin_Johnston_Saleh.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>
<br> <br> <br>

# Youtube Video Link
Check out our walkthrough video [here](https://youtube.com/watch?v=IXMQVvu-zYY)!

# Inquiries


# Acknowledgments
