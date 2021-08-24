
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as tfhub
from tensorflow.keras.models import Model
import tokenization as t

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

global bert_model

#Create a function to a load model
def load_model(model_path):
  """
  Loads a saved model from specified path.
  """
  print(f"Loading saved moodel from : {model_path}")
  # model=tf.keras.models.load_model(model_path, custom_objects={"KerasLayer":hub.KerasLayer})
  model=tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':tfhub.KerasLayer})

  return model

#load a trained  model 
bert_model=load_model("20210823-150923-Bert_ass.h5")


app = Flask(__name__)
@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['GET','POST'])
def my_form_post():

  text1 = request.form['text1'].lower()

  def replacehtml(x):
    x=x.replace("<+[\w+]+>","")
    return x
    
  max_seq_length = 60
  test_token = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="test_token")
  test_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="test_mask")
  test_segment = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="test_segment")

  #bert layer 
  bert_layer = tfhub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
  pooled_output, sequence_output = bert_layer([test_token, test_mask, test_segment])
  bert_model_machine = Model(inputs=[test_token, test_mask, test_segment], outputs=pooled_output)
  
  def test_tokens(text):
    T=[]
    M=[]
    S=[]
    tokens=tokenizer.tokenize(text)
    if len(tokens) > max_seq_length-2:
      tokens=tokens[0:(max_seq_length-2)]
    if len(tokens)< max_seq_length-2:
      dif=(max_seq_length-2)-len(tokens)
    for j in range(dif):
      tokens=[*tokens,'[PAD]']
    tokens=['[CLS]',*tokens,'[SEP]']
    #mask
    ms=[]
    mask=tokenizer.convert_tokens_to_ids(tokens)
    T.append(mask)
    for j in mask:
      if j>0:
        ms.append(1)
      else:
        ms.append(0)
    M.append(ms)
        
      # segment
    z=[]
    S.append(np.zeros(max_seq_length))
    return np.asarray(T,dtype=float),np.asarray(M,dtype=float),np.asarray(S)

  vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
  do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
  print("Hi")
  tokenizer=t.FullTokenizer(vocab_file,do_lower_case)
  print(tokenizer)
  preprocess_text=replacehtml(text1)
  test_token, test_mask ,test_segment=test_tokens(preprocess_text)
  X_test_pooled_output=bert_model_machine.predict([test_token, test_mask ,test_segment])
  bert_prediction=bert_model.predict(X_test_pooled_output)

  cls=bert_prediction.argmax(axis=-1)
  proab=bert_prediction[0][cls]

  if cls==1:
    result = f"The review is \n ***{text1}*** \n The model predcited that the review is POSITIVE and probablity is {proab} "
  else:
    result = f"The review is \n ***{text1}*** \n The model predcited that the review is Negative and probablity is {proab} "
    
  return render_template('form.html',prediction_text=result)


if __name__=="__main__":
    app.run(debug=True)
