from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,smoking_detection,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Smoking_Detection(request):
    if request.method == "POST":

        if request.method == "POST":

            CHASSIS_NO= request.POST.get('CHASSIS_NO')
            MODEL_YEAR= request.POST.get('MODEL_YEAR')
            USE_OF_VEHICLE= request.POST.get('USE_OF_VEHICLE')
            MODEL= request.POST.get('MODEL')
            MAKE= request.POST.get('MAKE')
            gender= request.POST.get('gender')
            age= request.POST.get('age')
            height_cm= request.POST.get('height_cm')
            weight_kg= request.POST.get('weight_kg')
            waist_cm= request.POST.get('waist_cm')
            eyesight_left= request.POST.get('eyesight_left')
            eyesight_right= request.POST.get('eyesight_right')
            hearing_left= request.POST.get('hearing_left')
            hearing_right= request.POST.get('hearing_right')

        df = pd.read_csv('Smoking_Datasets.csv', encoding='latin-1')

        def apply_response(label):
            if (label == 0):
                return 0  # No Smoking
            elif (label == 1):
                return 1  # Smoking

        df['Label'] = df['smoking'].apply(apply_response)

        cv = CountVectorizer()
        X = df['CHASSIS_NO'].apply(str)
        y = df['Label']

        print("CHASSIS_NO")
        print(X)
        print("Results")
        print(y)

        X = cv.fit_transform(X)

        #X = cv.fit_transform(df['CHASSIS_NO'].apply(lambda x: np.str_(X)))

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB

        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print("ACCURACY")
        print(naivebayes)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_nb))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm

        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print("ACCURACY")
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression

        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Gradient Boosting Classifier")

        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
            X_train,
            y_train)
        clfpredict = clf.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, clfpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, clfpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, clfpredict))
        models.append(('GradientBoostingClassifier', clf))


        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        CHASSIS_NO1 = [CHASSIS_NO]
        vector1 = cv.transform(CHASSIS_NO1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if (prediction == 0):
            val = 'No Smoking Detection'
        elif (prediction == 1):
            val = 'Smoking Detection'

        print(val)
        print(pred1)

        smoking_detection.objects.create(
        CHASSIS_NO=CHASSIS_NO,
        MODEL_YEAR=MODEL_YEAR,
        USE_OF_VEHICLE=USE_OF_VEHICLE,
        MODEL=MODEL,
        MAKE=MAKE,
        gender=gender,
        age=age,
        height_cm=height_cm,
        weight_kg=weight_kg,
        waist_cm=waist_cm,
        eyesight_left=eyesight_left,
        eyesight_right=eyesight_right,
        hearing_left=hearing_left,
        hearing_right=hearing_right,
        Prediction=val)

        return render(request, 'RUser/Predict_Smoking_Detection.html',{'objs': val})
    return render(request, 'RUser/Predict_Smoking_Detection.html')



