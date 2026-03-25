import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import gradio as gr

model_path = './cnn_neumonia.keras'
IMG_HEIGHT = 64
IMG_WIDTH = 64
# 1. Cargar el modelo
try:
    model = keras.models.load_model(model_path)
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    # Detener la ejecución si el modelo no se carga
    exit()

def preprocess_image_for_prediction(image: Image.Image, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    img = image.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) # Añadir dimensión de lote
    img_array = img_array / 255.0 # Normalizar a [0, 1]
    return img_array

def predict_pneumonia(image_path_or_object):
    if isinstance(image_path_or_object, str):
        # Si es una ruta, cargar la imagen
        image = Image.open(image_path_or_object)
    else:
        # Si ya es un objeto PIL Image (como en Gradio)
        image = image_path_or_object

    # Preprocesar la imagen
    processed_image = preprocess_image_for_prediction(image, target_size=(IMG_HEIGHT, IMG_WIDTH))

    # Realizar la predicción
    prediction = model.predict(processed_image, verbose=0) # verbose=0 para no imprimir el progreso

    # Interpretar la predicción
    probability = prediction[0][0]
    if probability > 0.5:
        message = f"¡Tiene Neumonía! (Probabilidad: {probability:.2f})"
    else:
        message = f"No tiene Neumonía (Probabilidad: {probability:.2f})"

    return message

iface = gr.Interface(
    fn=predict_pneumonia,
    title="Predicción de Neumonía por Radiografía",
    description="Carga una imagen de radiografía y predice si contiene neumonía.",
    inputs=gr.Image(type="pil",label="Sube una imagen de rayos X de tórax"),
    outputs=gr.Textbox(label="Diagnostico"),
    examples=[
        './normal1.jpeg',
        './pneumonia1.jpeg'
    ]
)

iface.launch(debug=True,share=True)