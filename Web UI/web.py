import numpy as np
import os

from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

dic = {0 : 'Adulterated', 1 : 'Pure'}

# Step 1: Load the saved model
#loaded_model = load_model('/mnt/c/Users/Room201B/Documents/Salem - Thesis/Saved Models/Rice Bran/set3_rb_resnet_9.h5')
loaded_model = load_model('C:\\Users\\hp\\Documents\\VSU Files\\Fourth Year\\Thesis\\Program\\Saved Models\\Rice Bran\\set3_rb_resnet_9.h5')

loaded_model.make_predict_function()

def predict_label(img_path):
    img = image.load_img(img_path, target_size=(400, 400))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    prediction = loaded_model.predict(img_array)

    # For binary classification, convert probabilities to class labels
    if prediction > 0.5:  # Assuming 0.5 is the threshold
        predicted_class = 1  # Or whatever value represents "Pure" in your dic
    else:
        predicted_class = 0  # Or whatever value represents "Adulterated" in your dic

    return dic[predicted_class] # Use the dic to return the label string


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

# @app.route("/about")
# def about_page():
# 	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        filename = img.filename  # Get the original filename

        # Construct the path using os.path.join for cross-platform compatibility
        static_folder = os.path.join(app.root_path, 'static') # Get the absolute path to the static folder
        img_path = os.path.join(static_folder, filename)

        # Ensure the static directory exists (create it if it doesn't)
        os.makedirs(static_folder, exist_ok=True)  # This will create any necessary parent directories as well.

        img.save(img_path)

        p = predict_label(img_path)

    return render_template("index.html", prediction=p, img_path=filename) # Pass the filename, not the full path


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
