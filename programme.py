from tensorflow.keras.models import load_model 
import matplotlib.pyplot as plt 
import numpy as np 
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import os

model = load_model('model_final_save.h5') # Chargement du modèle

test_image = 'data/prediction/prediction.jpg' # Image à prédire
img = load_img(test_image) # Chargement de l'image
plt.imshow(img) # Affichage de l'image
img_tensor = np.array(img.resize((299, 299))) # Redimensionnement de l'image 
img_tensor = np.expand_dims(img_tensor, axis=0) # Ajout d'une dimension
img_tensor = preprocess_input(img_tensor) # Prétraitement de l'image 

def predict(model, img, class_names): # Fonction de prédiction
    preds = model.predict(img_tensor) # Prédiction
    pred_label =  class_names[np.argmax(preds)] # Récupération de la classe prédite
    return pred_label # Retour de la classe prédite

prediction = predict(model, img_tensor, class_names=os.listdir("data/train")) # Prédiction
 
print(f"Label prédit: {prediction}") # Affichage du label prédit