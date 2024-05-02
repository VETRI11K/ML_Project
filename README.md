# Fake News Detection

Domain : Data Science/Natural Language Processor


# Aim
The aim of this project is to develop a model that can classify news articles as real or fake based on their content.

# Dataset
The dataset used for this project is sourced from a CSV file named "FakeNews.csv". 
It contains information about news articles including the author, title, content, and label indicating whether the news is real or fake.

# Data Pre-processing
1. Loaded the dataset into a pandas DataFrame.
2. Checked the shape of the dataset and displayed the first 5 rows.
3. Identified and handled missing values by replacing them with empty strings.
4. Merged the author name and news title into a single column named "content".
5. Performed text pre-processing including:
  ! Converting text to lowercase.
  ! Removing non-alphabetic characters.
  ! Tokenization and stemming using the Porter Stemmer algorithm.
  ! Removal of stopwords using NLTK's English stopwords list.
6. Converted textual data into numerical data using TF-IDF vectorization.

# Model Training
1. Split the dataset into training and test sets with a test size of 20% using stratified sampling.
2. Initialized and trained a Logistic Regression model on the training data.

Model Evaluation
1. Evaluated the model's performance on both training and test datasets using accuracy score.
2. Generated predictions for a sample news article and interpreted the results.
3. Visualized the distribution of classes (real vs. fake) in the dataset using a count plot.
4. Plotted the confusion matrix to visualize the model's performance in classifying real and fake news.

let's calculate the accuracy score:
# Importing necessary libraries
from sklearn.metrics import accuracy_score

# Calculating accuracy score on training data
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of the training data:', training_data_accuracy)

# Calculating accuracy score on test data
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of the test data:', test_data_accuracy)


# Conclusion
The model achieved accuracy on the test data, indicating its effectiveness in classifying news articles. 
Further optimization and fine-tuning of the model could be explored to improve its performance.
