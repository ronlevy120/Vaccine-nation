from django.shortcuts import render
import numpy as np
from .forms import CreateNewList
from prep import preprocess_input
import pickle


model = pickle.load(open('mysite/main/model.pickle', 'rb'))


def home(response):
    return render(response, 'main/home.html', {})


def test(response):
    if response.method == "POST":
        form = CreateNewList(response.POST)
        if form.is_valid():
            print("FORM IS VALID")
            n = form.cleaned_data["name"]
            df = preprocess_input(n)
            preds = model.predict_proba(df)
            result = np.asarray([np.argmax(line) for line in preds])[0]
        else:
            print("NOT VALID")
        return render(response, 'main/test.html', {"form": form, "output": result})

    else:
        form = CreateNewList()
    return render(response, 'main/test.html', {"form": form})


def idea(response):
    return render(response, "main/idea.html", {})


def team(response):
    return render(response, "main/team.html", {})

