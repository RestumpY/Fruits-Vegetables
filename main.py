import streamlit as st
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup

model = load_model('modelFinal.h5')


labels = {0: 'Amande', 1: 'Pomme', 2: 'Banane', 3: 'Betterave',4: 'Poivron', 5: 'Chou', 6: 'Piment',
          7: 'Carotte', 8: 'Noix de cajou', 9: 'Chou-fleur', 10: 'Piment', 11: 'Maïs', 12: 'Concombre', 13: 'Aubergine',
          14: 'Figue', 15: 'Ail', 16: 'Gingembre', 17: 'Raisins', 18: 'Piment jalapeño',
          19: 'Kiwi', 20: 'Citron', 21: 'Laitue', 22: 'Mangue', 23: 'Oignon', 24: 'Orange', 25: 'Paprika',
          26: 'Poire', 27: 'Petit pois', 28: 'Ananas', 29: 'Grenade', 30: 'Pomme de terre', 31:'Radis', 32: 'Raisins secs',
          33: 'Soja', 34: 'Epinard', 35: 'Fraise', 36: 'Maïs doux' ,37:"Patate douce",38:"Tomate",39:"Navet",40:"Pastèque"}

fruits = ['Pomme', 'Banane', 'Poivron', 'Piment', 'Raisins', 'Piment jalapeño', 'Kiwi', 'Citron', 'Mangue', 'Orange','Piment', 'Poire', 'Ananas', 'Grenade', 'Pastèque', 'Fraise']

vegetables =  ['Betterave', 'Chou', 'Paprika', 'Carotte', 'Chou-fleur', 'Maïs', 'Concombre', 'Aubergine', 'Gingembre',
'Laitue', 'Oignon', 'Petit pois', 'Pomme de terre', 'Radis', 'Soja', 'Epinard', 'Maïs doux', 'Patate douce',
'Tomate', 'Navet']

fruits_sec=['Amande', 'Noix de cajou', 'Figue', 'Raisins secs']



def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()


def run():
    st.title("Fruits / Légumes ou Fruits secs ?!")
    img_file = st.file_uploader("Choisir une image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if img_file is not None:
            result = processed_img(save_image_path)
            print(result)
            
            if result in vegetables:
                st.info('**Catégorie : Légumes**')
            elif result in fruits:
                st.info('**Catégorie : Fruits**')
            elif result in fruits_sec:
                st.info('**Catégorie : Fruits secs**')
            st.success("**Prediction : " + result + '**')

run()
