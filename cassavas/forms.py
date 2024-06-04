# myapp/forms.py
from django import forms
from .models import CassavaImage

class CassavaImageForm(forms.ModelForm):
    class Meta:
        model = CassavaImage
        fields = ['image']
