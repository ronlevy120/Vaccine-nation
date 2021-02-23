from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .models import ToDoList, Item
from .forms import CreateNewList


def home(response):
    return render(response, 'main/home.html', {})


def test(response):
    if response.method == "POST":
        form = CreateNewList(response.POST)
        if form.is_valid():
            print("FORM IS VALID")
            n = form.cleaned_data["name"]
        else:
            print("NOT VALID")
        return render(response, 'main/test.html', {"form": form, "output": n})

    else:
        form = CreateNewList()
    return render(response, 'main/test.html', {"form": form})


def idea(response):
    return render(response, "main/idea.html", {})


def team(response):
    return render(response, "main/team.html", {})

