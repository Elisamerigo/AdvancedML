from PIL import Image
import os
from monai.data import Dataset, DataLoader
from main import load_model, preprocess
# 1. Definisci il percorso dell'immagine
# Assicurati che l'immagine sia nella stessa cartella del file .py
# oppure inserisci il percorso completo.
image_path = "/Users/elisamerigo/Desktop/AdvancedML/Chest_Xray_PA_3-8-2010.png" 

try:
    # 2. Apertura dell'immagine
    img = Image.open(image_path)
    
    # 3. Visualizzazione (apre il visualizzatore predefinito del sistema operativo)
    img.show()

    # Opzionale: Stampa informazioni sull'immagine
    print(f"Formato: {img.format}")
    print(f"Dimensioni: {img.size}")
    print(f"Modalità: {img.mode}")

except FileNotFoundError:
    print(f"Errore: Il file '{image_path}' non è stato trovato.")
    print(f"Controlla che il percorso sia corretto e che il file esista nella cartella: {os.getcwd()}")

model=load_model()

preprocessed_image = preprocess(img)
x = model(preprocessed_image)
Image.fromarray((x[0, 0].detach().numpy()).astype('uint8')).show()