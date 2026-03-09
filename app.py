from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

print("Loading models...")

skin_model = tf.keras.models.load_model("models/skin_model.h5")
heart_model = joblib.load("models/heart_model.pkl")

print("Models loaded")

with open("database/disease.json") as f:
    knowledge = json.load(f)

class_names = [
"acne",
"eczema",
"psoriasis",
"melanoma",
"vitiligo",
"rosacea",
"seborrheic dermatitis",
"warts",
"chickenpox",
"impetigo"
]

state="idle"
answers={}

@app.route("/")
def home():
    return render_template("index.html")

def skin_prediction(file):

    img=Image.open(file)
    img=img.resize((224,224))

    img=np.array(img)/255.0
    img=img.reshape(1,224,224,3)

    prediction=skin_model.predict(img)

    index=np.argmax(prediction)

    confidence=float(np.max(prediction))*100

    disease=class_names[index]

    info=knowledge.get(disease,{})

    return f"""
Disease: {disease}
Confidence: {confidence:.2f} %

Cause: {info.get('cause')}

Treatment: {info.get('treatment')}

Routine: {info.get('routine')}

Message: {info.get('message')}
"""

@app.route("/chat",methods=["POST"])
def chat():

    global state,answers

    message=request.form.get("message")

    if "image" in request.files:

        img=request.files["image"]

        result=skin_prediction(img)

        return jsonify({"reply":result})

    if state=="idle":

        if "heart" in message.lower() or "chest" in message.lower():

            state="age"

            return jsonify({"reply":"I can help check your heart health. How old are you?"})

        if "skin" in message.lower() or "rash" in message.lower():

            return jsonify({"reply":"Please upload a clear image of the affected skin area."})

        return jsonify({"reply":"You can describe symptoms or upload a skin image."})

    elif state=="age":

        answers["age"]=float(message)
        state="gender"

        return jsonify({"reply":"Are you male or female?"})

    elif state=="gender":

        answers["sex"]=1 if "male" in message.lower() else 0
        state="bp"

        return jsonify({"reply":"Do you know your blood pressure?"})

    elif state=="bp":

        answers["bp"]=float(message)
        state="chol"

        return jsonify({"reply":"What is your cholesterol level?"})

    elif state=="chol":

        answers["chol"]=float(message)

        features=[
            answers["age"],
            answers["sex"],
            2,
            answers["bp"],
            answers["chol"],
            0,1,150,0,1,2,0,2
        ]

        result=heart_model.predict([features])

        state="idle"

        disease="heart disease" if result[0]==1 else "normal"

        info=knowledge.get(disease,{})

        return jsonify({

        "reply":f"""
Disease: {disease}

Cause: {info.get('cause')}

Treatment: {info.get('treatment')}

Routine: {info.get('routine')}

Message: {info.get('message')}
"""
        })

if __name__=="__main__":
    app.run(debug=True)