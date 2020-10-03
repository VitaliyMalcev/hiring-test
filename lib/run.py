# coding: utf-8

import pandas as pd
from keras.models import load_model
from pymystem3 import Mystem
import numpy as np
from keras.preprocessing import sequence
import pickle

if __name__ == '__main__':
    
    test = pd.read_csv('/hiring-test-data/test.csv')
    model = load_model('model.h5') #загружаем модель
    with open('tokenizer.pickle', 'rb') as handle:  # загружаем обученный токенайзер
        tok = pickle.load(handle)	

    m = Mystem()
    def lemmatize(text):
        return m.lemmatize(text)
    test['description'] = test['description'].apply(lemmatize) #лематизируем
	
	
    max_len = 350
    test_sequences = tok.texts_to_sequences(test['description'])
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len) #строим матрицу
	
    preds = model.predict(test_sequences_matrix)#предсказываем
    
    target_prediction = pd.DataFrame()
    target_prediction['index'] = range(test.shape[0])
    target_prediction['prediction'] = preds

    target_prediction.to_csv('/hiring-test-data/prediction.csv', index=False)
