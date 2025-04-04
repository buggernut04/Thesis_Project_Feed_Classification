import numpy as np
import cv2
import os
import uuid
import flask
import urllib
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, send_file
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Dictionary to store loaded models
models = {}

# Load your models (replace with your actual paths)
model_paths = {
    "rb_model": 'C:\\Users\\hp\\Documents\\VSU Files\\Fourth Year\\Thesis\\Program\\Saved Models\\Final Models\\Rice Bran\\set3_rb_resnet_4.h5',
    "corn_model": 'C:\\Users\\hp\\Documents\\VSU Files\\Fourth Year\\Thesis\\Program\\Saved Models\\Final Models\\Corn\\set2_corn_resnet_5.h5', 
    "sbm_model": 'C:\\Users\\hp\\Documents\\VSU Files\\Fourth Year\\Thesis\\Program\\Saved Models\\Final Models\\Soybean Meal\\set2_sbm_custom_3.h5',
}

for model_name, path in model_paths.items():
    try:
        models[model_name] = load_model(path)
        print(f"Model '{model_name}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

def resize_img(img_samp, target_size):
    # Convert BGR to RGB for Matplotlib display
    img_rgb = cv2.cvtColor(img_samp, cv2.COLOR_BGR2RGB)

    #print(img_rgb.shape)

    t_size = target_size
    pos = 2

    # Define the rectangle coordinates
    x1 = (img_rgb.shape[1] - t_size) // pos
    y1 = (img_rgb.shape[0] - t_size) // pos
    x2 = (img_rgb.shape[1] + t_size) // pos
    y2 = (img_rgb.shape[0] + t_size) // pos

    # Extract the region of interest (ROI) within the rectangle
    roi = img_rgb[y1:y2, x1:x2]

    # Resize the ROI to 224x224
    resized_roi = cv2.resize(roi, (224, 224))

    return resized_roi

# Update the classes list for binary classification
classes = ['Adulterated', 'Pure']

def predict(filename, model):
    img = image.load_img(filename)
    img_array = image.img_to_array(img)

    # Check if the image is not 224x224
    # if img.size != (224, 224):  # img.size returns (width, height)
    #     print("Image is not 224x224. Resizing...")
    #     img = resize_img(img, target_size=2800)  # Resize the image to 224x224

    # Check if the image is not 224x224
    # if img_array.shape[0] != 224 or img_array.shape[1] != 224:
    #     # Resize the image to 224x224 using the resize_img function
    #     print("Image is not 224x224. Resizing...")
    #     img_array = resize_img(img_array, target_size=2800)
    
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array)

    prob_positive = result[0][0]  # Probability of the positive class (e.g., 'Pure')
    prob_negative = 1 - prob_positive  # Probability of the negative class (e.g., 'Adulterated')

    # Round probabilities to 2 decimal places
    prob_positive = round(prob_positive * 100, 2)
    prob_negative = round(prob_negative * 100, 2)

    # Return class labels and their corresponding probabilities
    return [classes[1], classes[0], [prob_positive], [prob_negative]]

@app.route('/')
def home():
    selected_model = request.args.get('model_select', 'rb_model')  # Default to 'rb_model' if not provided
    return render_template("index.html", selected_model=selected_model)

@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    target_img = os.path.join(app.root_path, 'static', 'images')
    os.makedirs(target_img, exist_ok=True)
    
    if request.method == 'POST':
        selected_model = request.form.get('model_select')  # Get selected model
        print(f"Selected Model: {selected_model}")  # Debugging statement
        
        if not selected_model:
            error = "No model selected."
            return render_template('index.html', error=error)
        
        if selected_model not in models:
            error = "Invalid model selected."
            return render_template('index.html', error=error)

        model = models[selected_model] # Use the selected model
        
        if request.files:
            file = request.files.get('file')

            if file and allowed_file(file.filename):
                filename = file.filename
                img_path = os.path.join(target_img, filename)
                file.save(img_path)
                img = filename

                class_result_positive, class_result_negative, prob_positive, prob_negative = predict(img_path, model)

                predictions = {
                    "class1": class_result_positive,  # Positive class (e.g., 'Pure')
                    "class2": class_result_negative,  # Negative class (e.g., 'Adulterated')
                    "prob1": prob_positive[0],  # Probability of the positive class
                    "prob2": prob_negative[0],
                    "selected_model": selected_model  # Probability of the negative class
                }

            else:
                error = "Please upload images of jpg, jpeg, and png extension only"

            if len(error) == 0:
                return render_template('success.html', predictions=predictions, img=img)
            else:
                return render_template('index.html', error=error)
            
    elif request.form:
            link = request.form.get('link')

            try:
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                output = open(img_path, "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result_positive, class_result_negative, prob_positive, prob_negative = predict(img_path, model)

                predictions = {
                    "class1": class_result_positive,  # Positive class (e.g., 'Pure')
                    "class2": class_result_negative,  # Negative class (e.g., 'Adulterated')
                    "prob1": prob_positive[0],  # Probability of the positive class
                    "prob2": prob_negative[0],  # Probability of the negative class
                    "selected_model": selected_model
                }

                print('hi')

            except Exception as e:
                print(str(e))
                error = 'This image from this site is not accessible or inappropriate input'

            if len(error) == 0:
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)

    else:
        return render_template('index.html')
    
if __name__ == "__main__":
    app.run(debug=True)