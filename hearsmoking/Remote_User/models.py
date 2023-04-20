from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender= models.CharField(max_length=30)
    address= models.CharField(max_length=30)


class smoking_detection(models.Model):


    CHASSIS_NO= models.CharField(max_length=300)
    MODEL_YEAR= models.CharField(max_length=300)
    USE_OF_VEHICLE= models.CharField(max_length=300)
    MODEL= models.CharField(max_length=300)
    MAKE= models.CharField(max_length=300)
    gender= models.CharField(max_length=300)
    age= models.CharField(max_length=300)
    height_cm= models.CharField(max_length=300)
    weight_kg= models.CharField(max_length=300)
    waist_cm= models.CharField(max_length=300)
    eyesight_left= models.CharField(max_length=300)
    eyesight_right= models.CharField(max_length=300)
    hearing_left= models.CharField(max_length=300)
    hearing_right= models.CharField(max_length=300)
    Prediction= models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



