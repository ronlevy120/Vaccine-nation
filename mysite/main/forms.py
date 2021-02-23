from django import forms


class CreateNewList(forms.Form):
    name = forms.TextField(label="Name", max_length=200, required=False)
    check = forms.BooleanField(required=False)
