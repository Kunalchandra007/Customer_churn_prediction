# Customer Churn Prediction using ANN

📌 Project Overview

This project builds an Artificial Neural Network (ANN) model to predict customer churn using a dataset containing customer subscription details. The model is trained on structured data after preprocessing categorical variables and standardizing numerical features.

📂 Dataset Information
datasetlink-https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

The dataset contains various customer-related features, such as:

Gender

Subscription Type

Contract Length

Monthly Charges

Total Charges

Tenure

Payment Method

Churn (Target variable: 1 = Churn, 0 = Not Churn)

🔧 Technologies Used

Python: Programming language used for the project.

Pandas: For data manipulation and preprocessing.

Scikit-Learn: For preprocessing, encoding categorical features, feature scaling, and train-test splitting.

TensorFlow & Keras: For building, training, and evaluating the artificial neural network (ANN) model.

NumPy: For numerical operations and saving the model scaler.

🛠️ Installation & Setup

Ensure you have Python installed, then install dependencies using:

pip install pandas scikit-learn tensorflow numpy

🏗️ Model Training

1️⃣ Load and Preprocess Data

Load the dataset using Pandas.

Encode categorical columns using LabelEncoder.

Split the dataset into input features and target variable.

Standardize numerical features using StandardScaler.

2️⃣ Build and Train ANN Model

Create a sequential ANN model using Keras.

Add dense layers with ReLU activation for hidden layers.

Use a sigmoid activation function for binary classification.

Compile the model with the Adam optimizer and binary cross-entropy loss.

Train the model using batch processing and multiple epochs.

3️⃣ Save the Trained Model

Save the trained model as an H5 file.

Save the scaler for later use in predictions.

📊 Model Performance Evaluation

Predict churn probabilities for the dataset.

Convert probabilities to binary labels using a threshold of 0.5.

Evaluate model accuracy using the accuracy score metric.

🚀 Future Improvements

Tune hyperparameters for better accuracy.

Use more advanced feature engineering techniques.

Implement cross-validation for robust performance evaluation.
