from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = tf.keras.models.load_model("model/street_cleanliness_model.h5")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/stats")
def stats():
    return render_template("stats.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    os.makedirs("uploads", exist_ok=True)

    filepath = os.path.join("uploads", file.filename)

    file.save(filepath)

    img = image.load_img(filepath, target_size=(128,128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    confidence = float(prediction[0][0])

    if confidence > 0.5:
        result = "Dirty Street"
        confidence_percent = confidence * 100

        report = """
        The uploaded street image appears dirty.
        Waste accumulation was detected.
        Municipal cleaning attention may be required.
        Immediate inspection is recommended.
        """
    else:
        result = "Clean Street"
        confidence_percent = (1 - confidence) * 100

        report = """
        The uploaded street image appears clean.
        No visible waste accumulation was detected.
        The area seems well maintained.
        Regular monitoring is recommended.
        """

    color = "green" if result == "Clean Street" else "red"

    return f"""
    <html>
    <body style="font-family:Arial;background:#f4f6f9;text-align:center;">

    <h1 style="background:#2c3e50;color:white;padding:15px;">
    AI Street Cleanliness Analyzer
    </h1>

    <div style="width:700px;margin:30px auto;background:white;padding:20px;border-radius:15px;box-shadow:0px 0px 10px gray;">

    <h2>Uploaded Image</h2>

    <img src="/uploads/{file.filename}"
         width="300"
         style="border-radius:10px;">

    <br><br>

    <h2 style="color:{color};">{result}</h2>

    <h3>Confidence: {confidence_percent:.2f}%</h3>

    <h2>AI Generated Municipal Report</h2>

    <p style="font-size:18px;line-height:1.8;">
    {report}
    </p>

    <br>

    <a href="/" style="background:#3498db;color:white;padding:10px 20px;text-decoration:none;border-radius:8px;">
    Try Another Image
    </a>

    &nbsp;&nbsp;

    <a href="/stats" style="background:#27ae60;color:white;padding:10px 20px;text-decoration:none;border-radius:8px;">
    View Statistics
    </a>

    </div>

    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)