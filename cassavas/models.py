# myapp/models.py
from django.db import models

class CassavaImage(models.Model):
    image = models.ImageField(upload_to='cassava/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
