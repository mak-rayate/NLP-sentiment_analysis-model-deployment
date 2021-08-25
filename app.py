import re
import contractions
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

path = 'LSTM_Model.h5'
model = keras.models.load_model(path)


app = Flask(__name__)
@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['GET','POST'])
def my_form_post():
    

  text1 = request.form['text1'].lower()

  def clean_sentence(data):
      data = re.sub('((www.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',data)
      data = re.sub('((@[^\s]+))',' ',data)
      data = contractions.fix(data)
      data = re.sub('[^a-zA-Z]',' ',data)
      return data

  def preprocess_data(text):
        vocabulary_size=7000
        onehot_representation=[one_hot(text, vocabulary_size)]
        sentence_length=60
        embedded_documents=pad_sequences(onehot_representation, padding='pre', maxlen=sentence_length)
        return np.array(embedded_documents)

    
  data = preprocess_data(text1)
  preds = model.predict(data)
  if preds > 0.5:
      result = f'Status: not depressed || probability : {preds}'
  else:
      result = f'Status:depressed || probability : {preds}'
    
  return render_template('form.html',prediction_text=result)


if __name__=="__main__":
    app.run(debug=True)
