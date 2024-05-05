# ****Self-Driving Car Simulator | Udacity****

## Project Description

The goal of the self-driving car project is to use a deep learning model trained on image 
data to develop an autonomous driving system. The model uses input images from a dashboard 
camera in an automobile to train it to anticipate steering angles. This project makes use of 
the Udacity Self-Driving Car Simulator, which allows for virtual environment testing of the model.

#### Files included
* model.py Contains the code for building, training and saving the model.
* drive.py Contains the code for running the car.
* model.h5 Contains the model weights.
* requirements.txt Contains the dependencies need to install for this project.
* TRACK1-TRAINING Contains recording trained images after driving the udacity car simulator.

#### Project Instructions

* Clone this repository git clone 
* Download the datasets (see below)
https://drive.google.com/drive/folders/1J3C1S59yioMBUrWwFK6F0nVuT3cG1nsp?usp=sharing.wdrd24567t
    Extract the folder and save in the same root directory of this project.
    
* Run pip install -r requirements.txt to install the dependencies with correct versions
* This project works best on python version 3.10.7. This might not work with python current version.
* Download the Udacity simulator exe file click the link and select the link _Windows 64_ _Linux_ _Mac_
    as per your OS https://github.com/udacity/self-driving-car-sim/tree/term2_collection
* Run the following command to train your model to drive:

#####     Training:

    python model.py --sources TRACK1-TRAINING --train-on-autonomous-center

##### *Driving

    python drive.py --file model.h5

Video for the recording of vehicle driving autonomously around the track for at least one full lap.
https://youtu.be/dsnxUwNJHkE

##### References: 
* Naoki Repositories https://github.com/naokishibuya/car-behavioral-cloning/tree/master
* Udacity Self Driving Car Simulator https://github.com/udacity/self-driving-car-sim/tree/term2_collection
* Behavioral Cloning Repositories https://github.com/udacity/CarND-Behavioral-Cloning-P3/tree/master 

