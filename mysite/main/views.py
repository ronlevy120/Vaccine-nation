from django.shortcuts import render
import numpy as np
from .forms import CreateNewList
from mysite.main.prep import preprocess_input
import pickle

model = pickle.load(open('xgboost.pickle', 'rb'))


def home(response):
    return render(response, 'main/home.html', {})


def test(response):
    if response.method == "POST":
        form = CreateNewList(response.POST)
        print(f"form: {form}")
        if form.is_valid():
            print("FORM IS VALID")
            n = form.cleaned_data["name"]
            print(f"n: {n}")
            result = n
            # df = preprocess_input(n)
            # print(f"df: {df}")
            # preds = model.predict_proba(df)
            # result = np.asarray([np.argmax(line) for line in preds])[0]
            print(f"result type: {type(result)}, result: {result}")
        else:
            print("NOT VALID")
            result = "NOT VALID"
        return render(response, 'main/test.html', {"form": form, "output": result})

    else:
        form = CreateNewList()
    return render(response, 'main/test.html', {"form": form})


def idea(response):
    return render(response, "main/idea.html", {})


def team(response):
    return render(response, "main/team.html", {})

