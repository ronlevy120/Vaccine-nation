from mysite.main.views import preprocess_input
import pickle
import numpy as np

model = pickle.load(open('xgboost.pkl', 'rb'))

def prep(n):
    df = preprocess_input(n)
    prediction = model.predict_proba(df)
    result = np.asarray([np.argmax(line) for line in prediction])[0]
    result_final = 'Anti-Vaccine' if result == 1 else 'Pro-Vaccine'
    preds = prediction[0, 1]
    return result_final


n = 'hhh'
print(prep(n))
