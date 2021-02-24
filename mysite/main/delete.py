from mysite.main.prep import preprocess_input
import pickle
import numpy as np

model = pickle.load(open('xgboost.pickle', 'rb'))

def prep(n):
    df = preprocess_input(n)
    preds = model.predict_proba(df)
    result = np.asarray([np.argmax(line) for line in preds])[0]
    return result


n = 'hhh'
print(prep(n))
