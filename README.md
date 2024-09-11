
# Hotel Reviews Sentiment Analysis

## Project Overview:
This project applies **machine learning** techniques, specifically the **Naive Bayes algorithm**, to classify hotel reviews as either **positive** or **negative** based on their textual content. By analyzing word frequencies and leveraging sentiment patterns, this model predicts the sentiment of reviews in the test dataset. The project extensively utilizes **Python**, regular expressions (**RegEx**), and probability-based sentiment analysis to provide insights into customer satisfaction.

## Steps Involved:

### 1. **Data Preprocessing using RegEx:**
   - The dataset includes both **positive** (`hotelPosT-train.txt`) and **negative** (`hotelNegT-train.txt`) reviews.
   - **RegEx** was used for data cleaning, including:
     - **Removing punctuation** (such as `,`, `.`, `!`, etc.) and special characters.
     - **Eliminating numbers** and irrelevant tokens from the text.
     - **Filtering out common stop words** (like "and", "the", "in", etc.) to focus on meaningful words.
   - This preprocessing step ensures that only relevant information is passed to the machine learning model for sentiment analysis.

### 2. **Word Frequency and Probability Calculation:**
   - Word counts for both positive and negative reviews were calculated, stored in dictionaries, and processed to form the basis for the Naive Bayes classification.
   - For both positive and negative reviews:
     - Word frequencies were stored in dictionaries (`Positive_Word_Dict` and `Negative_Word_Dict`).
     - Maximum Likelihood Estimation (MLE) with **Laplace Smoothing** was used to calculate word probabilities.
     - The log probabilities of each word occurring in positive or negative reviews were computed to prevent numerical underflow.

### 3. **Machine Learning: Naive Bayes Classification:**
   - The **Naive Bayes algorithm** was employed to classify reviews based on word probabilities.
   - **Log Prior Calculation:** The prior probabilities for both positive and negative reviews were determined based on the ratio of positive and negative reviews in the dataset.
   - **Log Likelihood Calculation:** The likelihood for each word appearing in a review (whether positive or negative) was calculated using the word counts.
   - For each test review:
     - The sum of the log probabilities for words in the review was computed for both positive and negative classes.
     - The sentiment (positive or negative) was determined by comparing the log probabilities, and the review was classified accordingly.

### 4. **Sentiment Classification of Test Reviews:**
   - Reviews from the test set (`TestSet.txt`) were classified based on the calculated probabilities.
   - The output is a text file (`output.txt`), where each review ID is listed alongside its predicted sentiment (either positive or negative).

## Key Components and Techniques:
- **Naive Bayes Classifier (Machine Learning):** Utilized to classify reviews by learning from word frequencies in positive and negative reviews.
- **Regular Expressions (RegEx):** Used extensively for text preprocessing, including removing punctuation, numbers, and irrelevant tokens.
- **Text Preprocessing and Tokenization:** Cleaned and tokenized the text, ensuring only meaningful words were included for sentiment classification.
- **Probability Calculation:** Implemented log probabilities and used Laplace Smoothing to handle words not seen in the training data, ensuring a smooth classification process.

## Output:
The final output is stored in `output.txt`, which contains the ID of each test review along with its predicted sentiment (positive or negative).

## Tools & Libraries Used:
- **Python**: Core language used for implementing the Naive Bayes algorithm, RegEx for text cleaning, and file handling.
- **NumPy**: For numerical operations and log probability calculations.
- **TextBlob**: For basic natural language processing (NLP) and classification tasks.
- **RegEx (Regular Expressions)**: For advanced text cleaning and tokenization.
- **Counter (from collections module)**: Used for word frequency analysis.

## Conclusion:
By using the Naive Bayes machine learning algorithm and leveraging text processing with **RegEx**, this project effectively classifies hotel reviews based on sentiment. This approach helps in automating sentiment analysis and provides valuable insights into customer feedback.
