import streamlit as st
import PIL as pl
from predict import predict_img
from PIL import Image, ImageOps



with st.container(): 
    # Liste des fichiers uploadés
    uploaded_files = st.file_uploader("Sélectionnez des fichiers", type=['png', 'jpeg', 'jpg'], accept_multiple_files=True)

    # Vérifier si des fichiers ont été uploadés
    if uploaded_files:
        # Compteur pour afficher 3 images par ligne
        col = 0
    
        col1, col2, col3 = st.columns(3, gap="small")
        cols = [col1, col2, col3]
        # Boucle à travers les fichiers uploadés
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            # Transform to gray (1 channel)
            img = ImageOps.grayscale(img)
            # Resize the img
            size = 500 , 500
            img = img.resize(size)
            # Afficher le nom du fichier
            if col >= 3:
                col1, col2, col3 = st.columns(3, gap="small")
                cols = [col1, col2, col3]
                col = 0
            with cols[col]:  
                names = uploaded_file.name.split(".")
    
                # Afficher l'image
                st.image(img, use_column_width="auto" , caption=names[0][:15]+"... ."+names[1] )
                prediction = predict_img(uploaded_file)
                prediction =  str(prediction).split("'")
                st.write("Prediction du modèle est:")
                st.write(prediction[1])
                # Incrémenter le compteur
            col += 1
        
        
