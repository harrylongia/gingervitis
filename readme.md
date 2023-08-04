# Gingervitis Medical Diagnosis using Machine Learning

Gingervitis is a medical diagnosis project aimed at developing a machine-learning model to assist in diagnosing various medical conditions based on patient symptoms. The project uses a dataset containing symptom profiles and corresponding medical conditions to train and evaluate the machine-learning models.

## Dataset

For this project, I have used a Kaggle dataset containing symptom profiles and corresponding medical conditions. Each row in the dataset represents a patient's symptom profile, and the goal is to predict the diagnosed medical condition (prognosis) based on these symptoms.

The dataset is split into a training set and a test set, containing a total of 4920 samples.

For more details on the dataset, check it on Kaggle:

[https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)


## Inspiration

The Gingervitis Medical Diagnosis project is a result of my exploration and learning in machine learning. Drawing inspiration from various sources, I have developed machine-learning models for medical diagnosis. This project allowed me to build upon existing models and apply my understanding of machine learning algorithms to predict medical conditions based on patient symptoms. I am excited about the potential of this project to contribute to the improvement of healthcare through technology. 

## Model Training and Evaluation

Three different machine learning models were trained and evaluated on the dataset:

1. Random Forest Classifier
2. Naive Bayes Classifier (GaussianNB)
3. Support Vector Machine (SVM) Classifier

Each model was trained using the symptom features and the corresponding medical condition as the target variable. After training, the models were evaluated on a separate test set to assess their performance.

## Results

The accuracy of the models on the test set was as follows:

- Random Forest Classifier: 100%
- Naive Bayes Classifier: 100%
- Support Vector Machine (SVM) Classifier: 100%

## Setup and Usage

To use the trained models for medical diagnosis in the Gingervitis project, follow these steps:

1. **Clone the repository to your local machine.**

2. **Create a virtual environment (venv) and activate it.** This step is essential to isolate the project dependencies from the system-wide Python installation.

   - **Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

   - **Linux/Mac (Bash):**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install the required dependencies** from the 'requirements.txt' file using the following command:
   ```bash
   pip install -r requirements.txt

4. Launch Jupyter Notebook and open the provided '.ipynb' file. and run all the cells.
## Note

- The dataset used for training this model is for demonstration purposes only and may not be comprehensive or fully representative of real-world medical conditions.
- The project may produce incorrect answers in some cases due to the limited number of symptoms considered for diagnosis. The models' accuracy heavily relies on the presence of specific symptoms in the dataset, and missing symptoms could lead to erroneous predictions.
  

## Future Scope
- To improve the accuracy and robustness of the models, one area of improvement would be to gather a more extensive dataset with a broader range of medical conditions and their associated symptoms. This would allow the models to learn from a more diverse set of examples and make more reliable predictions.

- Additionally, better testing and validation procedures are essential to thoroughly assess the models' performance. Employing cross-validation techniques and testing on independent datasets can provide a more accurate estimate of how the models will perform in real-world scenarios.

- Tweaking the model's hyperparameters and considering the weighted contributions of different symptoms could also enhance its performance. Fine-tuning the models with a deeper understanding of the medical domain may lead to more accurate and reliable predictions.
  
Please enjoy this Gingervitis Medical Diagnosis project responsibly and with an open mind to learning and improvement. Thank you for joining me on this journey!

