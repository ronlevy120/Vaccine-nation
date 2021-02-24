from mysite.main.views import preprocess_input
import pickle
import numpy as np

model = pickle.load(open('xgboost.pkl', 'rb'))

def prep(n):
    df = preprocess_input(n)
    preds = model.predict_proba(df)
    result = np.asarray([np.argmax(line) for line in preds])[0]
    return preds[0, 1]


n = 'hhh'
print(prep(n))
