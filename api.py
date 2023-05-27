import base64
from io import BytesIO
from flask import Flask, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the TFLite model and define the preprocess_image function
interpreter = tf.lite.Interpreter(model_path='D:\\Android\\Face Recognition System\\model_3.tflite')
interpreter.allocate_tensors()

@app.route('/predict', methods=['POST'])
def predict():
    # Get the base64-encoded image from the request
    image_data = request.json['image']

    # Decode the base64-encoded image
    image_bytes = base64.b64decode(image_data)

    # Load and preprocess the input image
    image = Image.open(BytesIO(image_bytes))
    image = image.resize((224, 224))
    # Convert the image to a NumPy array
    image_array = np.array(image)
    # Preprocess the image (e.g., normalization, scaling) if required
    # Convert the data type to FLOAT32
    image_array = image_array.astype(np.float32)
    # ...
    image_array = np.expand_dims(image_array, axis=0)

    # Set the input tensor of the TFLite model
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # Run the inference
    interpreter.invoke()

    # Get the output tensor
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)


    # Process the output data to find the index of the class with the maximum probability
    predicted_class_index = np.argmax(output_data)
    print(predicted_class_index)

    # Return the predicted class and probability
    return {
        'class': predicted_class_index,
        'probability': output_data[predicted_class_index]
    }

@app.route('/call', methods=['GET'])
def call():
    return {'hello': 'user'}

if __name__ == '__main__':
    app.run(host ='0.0.0.0',port = 5000, debug=True)
