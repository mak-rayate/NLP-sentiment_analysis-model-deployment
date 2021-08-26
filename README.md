# NLP : Detection of depression and mental illness in Twitter using LSTM + Flask deployment + HEROKU deployment


#### Deployment Link : https://nlplstm.herokuapp.com/

![image](https://user-images.githubusercontent.com/67750027/130950904-e097af26-a62e-46bb-89e4-03e979805ad3.png)

Automatic text classification can be done in many different ways in machine learning as we have seen before.
This project aims to provide an example of how a Recurrent Neural Network (RNN) using the Long Short Term Memory (LSTM) architecture can be implemented using Keras. I have used the same data source as we used in Text Classification using ![BERT_model](https://github.com/mak-rayate/NLP-Sentiment_Analysis-using-BERT). If you have checked ![BERT_model](https://github.com/mak-rayate/NLP-Sentiment_Analysis-using-BERT) is excellent on data where as LSTM has some limitation. 
#### This repository is more focusing on model deployment task on Public URL.

#### The Data :
      <class 'pandas.core.frame.DataFrame'>
      RangeIndex: 1600000 entries, 0 to 1599999
      Data columns (total 2 columns):
       #   Column  Non-Null Count    Dtype 
      ---  ------  --------------    ----- 
       0   target  1600000 non-null  int32 
       1   text    1600000 non-null  object
      dtypes: int32(1), object(1)
      memory usage: 18.3+ MB
      
      
### Clasification Report
                     precision    recall  f1-score   support

                 0       0.83      0.81      0.82    159815
                 1       0.81      0.83      0.82    160185

          accuracy                           0.82    320000
         macro avg       0.82      0.82      0.82    320000
      weighted avg       0.82      0.82      0.82    320000
      
### Output : ----------------------------------------------------------------------------------------------

##### input statement : "The Food is Excellent.The food is excellent, generous portions and great prices"

![Screenshot (523)](https://user-images.githubusercontent.com/67750027/130953326-361a31d4-7d34-4c8e-8598-a173f8b8ce6d.png)

