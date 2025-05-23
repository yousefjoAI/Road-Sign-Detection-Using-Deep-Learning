import os
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model #type: ignore
from PIL import Image

# Label Overview
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Veh > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing veh > 3.5 tons' }

app = Flask(__name__)

# Allowed image file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

# Define available models
MODELS = {
    "Custom CNN 1": "custom_cnn_1.h5",
    "Custom CNN 2": "custom_cnn_2.h5",
    "VGG19": "vgg19.h5",
    "ResNet50": "resnet.h5",
    "Ensemble Predictions 1": "ensemble",
    "Ensemble Predictions 2" : "ensemble_2"
}

details = {
    'Custom CNN 1': 'Precision: 0.95\nRecall: 0.96\nF1 Score: 0.96\nAccuracy: 97%\n',
    'Custom CNN 2': 'Precision: 0.94\nRecall: 0.95\nF1 Score: 0.94\nAccuracy: 96%\n',
    'VGG19': 'Precision: 0.90\nRecall: 0.91\nF1 Score: 0.90\nAccuracy: 92%\n',
    'ResNet50': 'Precision: 0.76\nRecall: 0.75\nF1 Score: 0.75\nAccuracy: 80%\n',
    'Ensemble Predictions 1': 'This combines the best two models to predict the sign.\nCustom CNN 1 + Custom CNN 2',
    'Ensemble Predictions 2' : 'This combines the best three models to predict the sign.\nCustom CNN 1 + Custom CNN 2 + VGG19'
}


# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Pre-load models
loaded_models = {}
for model_name, model_path in MODELS.items():
    if model_path not in ['ensemble', 'ensemble_2']:
        loaded_models[model_name] = load_model(model_path)

# Function to check if the file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found at the specified path.")
    
    # Convert to RGB and resize
    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((32, 32))

    # Convert to numpy array and normalize
    image_array = np.array(resize_image) / 255.0
    return np.expand_dims(image_array, axis=0), None

# Ensemble helper (soft-voting)
def ensemble_predictions(image: np.ndarray, version):
    if version == 1:
        preds = [loaded_models[name].predict(image) for name in ['Custom CNN 1', 'Custom CNN 2']]
    else:
        preds = [loaded_models[name].predict(image) for name in ['Custom CNN 1', 'Custom CNN 2', 'VGG19']]
    avg = sum(preds) / len(preds)
    return avg

@app.route("/")
def index():
    return render_template("index.html", models=MODELS.keys(), model_details=details)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"})

    file = request.files["file"]
    selected_model = request.form.get("model")

    if file.filename == "":
        return jsonify({"error": "No file selected!"})

    if file and allowed_file(file.filename):
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Preprocess the image
        image, error = preprocess_image(file_path)
        if error:
            return jsonify({"error": error})
        
        if selected_model == 'Ensemble Predictions 1':
            probs = ensemble_predictions(image, 1)
        elif selected_model == 'Ensemble Predictions 2':
            probs = ensemble_predictions(image, 2)
        else:
            model = loaded_models[selected_model]
            probs = model.predict(image)

        # Predict
        predicted_class = np.argmax(probs)
        confidence = np.max(probs) * 100
        meta_image_path = f"images/{predicted_class}.png"
        print(meta_image_path)

        # Delete the uploaded file after processing
        os.remove(file_path)

        return jsonify({
            "class": classes.get(predicted_class, "Unknown"),
            "confidence": f"{confidence:.2f}%",
            "meta_image": meta_image_path,
        })

if __name__ == "__main__":
    app.run(debug=True)