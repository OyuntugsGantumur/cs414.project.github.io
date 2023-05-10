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
In this project, we are aiming to create a machine learning model that can recognize and identify emotions based on facial expressions. Human communication is composed of verbal, such as language use, and nonverbal cues, including body language, speech tone, and facial expressions. Most of the time, we express our current emotions and feelings through our facial expressions and we also read other people's affective tones through their facial expressions and modify our way of communicating with them. For example, I would never ask my mom for anything if I see her eyebrows frowned, which means she's either angry or so concentrated. So recognizing emotions and identifying them corretly is an important aspect of our communication. 

We believe emotion recognition has a variety of real-life applications and future implications in our day-to-day lives. First, a successful model could help people, especially those with developmental disability or phychopathic traits that affect their empathetic skills, recognize emotions in their daily communication and possibly increase the quality of their life (Kyranides et al., 2022). The more elaborate version of the model could even be used in increasing road safety. A research has shown that angry drivers take more risks on the road and have more car accidents than low-anger drivers (Parkinson, 2001). Emotion recognition models in combination with tiredness and attention could be beneficial to driving safety and have implications for car manufacturers. Lastly, emotion recognition could largely impact the human-machine interactions. Machines could possibly respons differently based on our emotions and mood just as we do in our human-to-human communications. It is controversial if machines should be able to understand human feelings, but it would be another big step forward in human-machine interaction. 

In this final project, we wanted to explore the capabilities of machine learning in the domain of face detection and emotion recognition. Our goal was to train a model using a large, publicly available dataset and then apply our model on handpicked real-life images that contain multiple people in various environments and backgrounds. We conducted this experiment to observe how the model would act under a new environment and how accurate the results would be.


# Data Gathering
(SHAKH)
- Usage of Kaggle dataset
- image of the kaggle dataset interaction and display



# Model Structure
(SHAKH)
## Base Model
- Usage of existing code and its results/accuracy


## Modifications
- Our modificaions (batch normalization + maxpooling2d)
- Runtime issues and other errors, trial and error process



# Facial Recognition

After we trained our model, we wanted to apply the model in recognizing emotions in real-life images with multiple people in various scenarios. To do so, we executed the following steps:

1. Converting the image to grayscale
2. Implement pre-trained haarcascade classifier for face detection
3. Tune the hyperparameters of haarcascade
4. Predict the emotions

Let's use the below image as example for how we applied our model in "real-life" images ![alt text](https://github.com/OyuntugsGantumur/ML_project/blob/main/test_images/test_0.jpg?raw=true)


1. Before performing any face detection, we first converted the image to grayscale to reduce its noise and improve computational efficiency as it is easier to identify faces in grayscale than in color. 


For face detection, we used the pre-trained haarcascade classifier built in OpenCV. This algorithm uses edge or line detection features proposed by Viola and Jones. This model is based on the sum of pixels for various facial features - such as eyes are darker than the nose and cheeks regions, and eyes are darker than the bridge of the nose - and is trained with a lot of positive images with faces and negative images with no face. We chose to use the OpenCV haarcascade classifier because it is fast and requires less computing power, making it suitable for our limited time and resource. It can also detect faces in a wide range of orientations and scales, making them versatile for a variety of situations and scenarios. However, we have later learned that the downside of the haarcascade model is that it produces many false positives. In order to reduce the number of false positive, we decided to make the filtering process a bit more strict: we increased the minNeighbors to three so false positives from the background are removed; and we modified the scaleFactor to 1.3 so the larger faces, or the main faces in the frame, can be detected faster and more accurately. These changes run the risk of missing some faces, but we think eliminating false positives is more important than missing a few true positives in terms of efficiency and accuracy.

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


# References

Kyranides, M.N., Christofides, D. & Çetin, M. Difficulties in facial emotion recognition: taking psychopathic and alexithymic traits into account. BMC Psychol 10, 239 (2022). https://doi.org/10.1186/s40359-022-00946-x

Parkinson, B. (2001). Anger on and off the road. British journal of Psychology, 92(3), 507-526.