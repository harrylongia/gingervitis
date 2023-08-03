# Gingervitis Medical Diagnosis using Machine Learning

Gingervitis is a medical diagnosis project aimed at developing a machine learning model to assist in diagnosing various medical conditions based on patient symptoms. The project uses a dataset containing symptom profiles and corresponding medical conditions to train and evaluate the machine learning models.

## Dataset

For this project, I have used a Kaggle dataset containing symptom profiles and corresponding medical conditions. Each row in the dataset represents a patient's symptom profile, and the goal is to predict the diagnosed medical condition (prognosis) based on these symptoms.

The dataset is split into a training set and a test set, containing a total of 4920 samples.

For more details on the dataset, check the it on kaggle:

[https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)


## Inspiration

The Gingervitis Medical Diagnosis project is a result of my exploration and learning in the field of machine learning. Drawing inspiration from various sources, I have developed machine learning models for medical diagnosis. This project allowed me to build upon existing models and apply my understanding of machine learning algorithms to predict medical conditions based on patient symptoms. I am excited about the potential of this project to contribute to the improvement of healthcare through technology. 

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
- Medical diagnosis should always be conducted by qualified medical professionals, and machine learning models should be used as supportive tools and not as substitutes for medical expertise.

Please enjoy this Gingervitis Medical Diagnosis project responsibly and with an open mind to learning and improvement. Thank you for joining me on this journey!

