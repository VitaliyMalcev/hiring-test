# coding: utf-8

import pandas as pd
from keras.models import load_model
from pymystem3 import Mystem
import numpy as np



if __name__ == '__main__':
    
    test = pd.read_csv('/hiring-test-data/test.csv')
    model = load_model('/model.h5') 
    
    for i in range(10):
        test[str(i) +" count"] = test['description'].apply(lambda x: x.count(str(i)))

    m = Mystem()
    def lemmatize(text):
        return m.lemmatize(text)

    test['description'] = test['description'].apply(lemmatize)

    words = ['один', 'два', 'три', 'четыре', 'пять','шесть',"семь", 'восемь', 'девять', 'ноль']
    for i in words:
        test[str(i) +" count"] = test['description'].apply(lambda x: x.count(str(i)))


    test_features = test.drop(['title','description','subcategory', 'category', 'price', 'region', 'city', 'datetime_submitted'],axis =1)
	
    preds = model.predict(test_features)
    
    target_prediction = pd.DataFrame()
    target_prediction['index'] = range(test.shape[0])
    target_prediction['prediction'] = preds

    target_prediction.to_csv('/hiring-test-data/prediction.csv', index=False)
