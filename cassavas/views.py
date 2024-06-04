import os
import logging
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import Model # type: ignore
import numpy as np
import cv2
from PIL import Image
from .forms import CassavaImageForm

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load your model
model_path = r'C:\Users\hp\Desktop\Cassava_web\cassava\cassavas\static\resnet_weight_TB.h5'
model = load_model(model_path)

def preprocess_image(image_path):
    img = Image.open(image_path)
    # Crop the image to the center
    width, height = img.size
    new_width, new_height = 112, 112

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    img = img.crop((left, top, right, bottom))
    img = img.resize((112, 112))  # Ensure the image is resized to the correct input size for the model
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = np.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, superimposed_img)

def get_solution(predicted_class):
    solutions = {
        "Bacterial Blight": [
            "Use Disease-Free Planting Material: Start with healthy, disease-free planting materials to minimize the chances of introducing the disease to new areas.",
            "Crop Rotation: Practice crop rotation to break the disease cycle. Avoid planting cassava or related crops in the same field for consecutive seasons.",
            "Resistant Varieties: Plant resistant cassava varieties that have shown resistance to CBSD and CMD. Utilize improved varieties that have been bred to withstand the disease.",
            "Early Detection and Removal: Regularly monitor your cassava plants for any signs of disease. If infected, remove and destroy the affected plants to prevent the spread of the disease.",
            "Proper Planting Density: Avoid planting cassava too closely, as this can promote the spread of diseases due to increased humidity and reduced airflow.",
            "Pruning and Thinning: Regularly prune and thin cassava plants to remove infected and weak parts, improving overall plant health."
        ],
        "brown spot": [
            "Planting cuttings from disease-free areas/gardens after being certified by MAAIF",
            "Planting tolerant varieties e.g. NASE 14, 19, and NAROCAS1. (Farmers should from time to time seek advice from Extension staff on the latest tolerant varieties)",
            "Uprooting and burying or burning diseased plants or entire field.",
            "Integrated disease management.",
            "Most CMD resistant/tolerant varieties are susceptible to CBSD but the tolerant varieties currently recommended to farmers include NASE 14, 19 and NAROCAS1."
        ],
        "green mite": [
            "To avoid this pest, make sure to plant early at the onset of rains.",
            "Biological control through using predatory mites and parasites.",
            "Carry out crop rotation.",
            "Integrated Pest Management is also highly advisable."
        ],
        "Mosaic": [
            "Plant resistant or tolerant varieties e.g. the NASE and TME varieties (e.g. NASE 14, 15, 16, 17, 18, 19, NAROCAS 1, 2 and TME 14 and 204)",
            "Practice crop rotation with legumes",
            "Use disease-free planting material"
        ]
    }
    return solutions.get(predicted_class, [])

def predict(request):
    class_labels = ["Bacterial Blight", "Brown Spot", "Green Mite", "Mosaic"]

    if request.method == 'POST' and request.FILES.get('cassava_image'):
        cassava_image = request.FILES['cassava_image']
        image_path = default_storage.save(cassava_image.name, cassava_image)
        image_path = default_storage.path(image_path)
        img_array = preprocess_image(image_path)

        # Make prediction
        prediction = model.predict(img_array)
        logging.info(f"Predictions: {prediction}")

        # Check the shape of the prediction array
        logging.info(f"Prediction Shape: {prediction.shape}")

        # If the prediction array has multiple samples, select the index with the highest confidence for each sample
        predicted_class_indices = np.argmax(prediction, axis=1)
        logging.info(f"Predicted Class Indices: {predicted_class_indices}")

        # Get the predicted classes
        predicted_classes = [class_labels[idx] for idx in predicted_class_indices]
        logging.info(f"Predicted Classes: {predicted_classes}")

        # Generate Grad-CAM heatmap
        last_conv_layer_name = 'act_last'  # Adjust based on your model's architecture
        heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=predicted_class_indices[0])

        # Save Grad-CAM
        explanation_path = os.path.join(settings.MEDIA_ROOT, 'gradcam', os.path.basename(image_path).replace('.jpg', '_gradcam.jpg'))
        os.makedirs(os.path.dirname(explanation_path), exist_ok=True)
        save_and_display_gradcam(image_path, heatmap, cam_path=explanation_path)

        # Get the solution based on the prediction
        solutions = get_solution(predicted_classes[0])


        return render(request, 'cassavas/result.html', {
            'image_path': default_storage.url(image_path),
            'prediction': predicted_classes[0],
            'explanation_path': default_storage.url(explanation_path),
            'solutions': solutions,
        })

    else:
        form = CassavaImageForm()
    return render(request, 'cassavas/index.html', {'form': form})

def upload_image(request):
    if request.method == 'POST':
        uploaded_image = request.FILES['image']
        return render(request, 'cassavas/upload_success.html', {'uploaded_image': uploaded_image})
    else:
        return render(request, 'cassavas/index.html')
