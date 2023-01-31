import plotly.graph_objects as go
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st 
import pandas as pd
import numpy as np
import glob
import cv2

from Filtering import filter_Kmeans1, filter_Kmeans2, filter_KmeansXYRGB, EqualizerImg, filter_MeanShift
from plotly.subplots import make_subplots
from keras.preprocessing import image
from PIL import Image


#####------------------------------FONCTIONS------------------------------#####
path_resources = "resources/"
path_models = "models/"

# Fonction pour charger les données des modèles et afficher les resources (matrice, rapport et courbes(loss et accuracy)
def load_resources(history_name, report_name, matrix_name):
    
    st.markdown("<h2 style='text-align: left; color: #00008B;'>Classification scores</h2>", unsafe_allow_html=True)
    st.info("###### In this part, you can see the results of your selections. These graphs below show the performance of the model in terms of the filter selected")

    h_loaded = np.load(path_resources + history_name, allow_pickle='TRUE').item() # Chargement de l'historique
    r_loaded = pd.read_csv(path_resources + report_name,index_col=0).astype(str).replace("nan", "") # Chargement du rapport
    r_loaded.index = ['neutrophil', 'eosinophil', 'ig','platelet','erythroblast','monocyte','basophil','lymphocyte','accuracy','macro avg','weighted avg']

    
    col_left, col_right = st.columns((1, 1))
    with col_right: 
        st.markdown("<h5 style='text-align: center; color: black;'>Confusion matrix</h5>", unsafe_allow_html=True)

        st.image(path_resources + matrix_name,width= 450) # Affichage de la matrice de confusion
    with col_left:
        st.markdown("<h5 style='text-align: center; color: black;'>Classification report</h5>", unsafe_allow_html=True)
        st.dataframe(r_loaded, height=340) # Affichage du rapport de classification
    
    st.markdown("<h5 style='text-align: center; color: black;'>Accuracy & loss along epochs</h5>", unsafe_allow_html=True)
    
    col_left2,col_center,col_right2 = st.columns((0.1,1, 0.01))
    with col_center:
        # Affichage des courbes loss et accuracy du modèle
        epochs = np.arange(1 , len(h_loaded["loss"])+1, 1)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(x=epochs, y=h_loaded['accuracy'],mode='lines',name='accuracy'
                                ,line=dict(color='royalblue', width=3)),secondary_y=False)
        fig.add_trace(go.Scatter(x=epochs, y=h_loaded['val_accuracy'],mode='lines',name='val_accuracy'
                                ,line=dict(color='firebrick', width=3)),secondary_y=False)
        fig.add_trace(go.Scatter(x=epochs, y=h_loaded['loss'],mode='lines',name='loss'
                                ,line=dict(color='royalblue', width=3, dash = 'dot')),secondary_y=True)
        fig.add_trace(go.Scatter(x=epochs, y=h_loaded['val_loss'],mode='lines',name='val_loss'
                                ,line=dict(color='firebrick', width=3, dash = 'dot')),secondary_y=True)
        fig.update_layout(width = 1000,margin=dict(l=0,r=0,b=0,t=0,pad=4)
                        ,font=dict(size=14)
                        ,legend = dict(font = dict(size = 14, color = "black")))
        fig.update_yaxes(title_text="<b>Accuracy</b> scale", secondary_y=False)
        fig.update_yaxes(title_text="<b>Loss</b> scale", secondary_y=True)
        st.plotly_chart(fig)

def ma_fonction_dl():
    st.markdown("<h1 style='text-align: left; color: #A52A2A;'>Deep Learning</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: left; color: #00008B;'>Model and filter selection</h2>", unsafe_allow_html=True)
    st.info("###### First, select a deep learning model. An overview of the architecture model is displayed next to model selection")
    col1, col2 = st.columns((0.2,1))

    st.info("###### Then, choose a filter image and an overview of the filter is displayed next to the filter selection.")
    col3, col4 = st.columns((0.3,1))

    with col1:
        st.markdown("  ")
        st.markdown("  ")
        st.markdown("  ")
        st.markdown("  ")
        st.markdown("  ")

        option_model_dl = ['CNN (From Scratch)','VGG16','VGG19','ResNet50','Xception']
        o_model_dl = st.radio('Select model',option_model_dl)
    
    with col3:
        option_filter = ['Original (RGB)', 'Kmeans_1', 'Kmeans_2', 'Kmeans XYRGB', 'Equalizer', 'Mean Shift']
        o_filter = st.radio("Select filter",option_filter)
    
    if o_model_dl == 'CNN (From Scratch)' :
        with col2:
            st.image(path_resources + "model_cnn_fs.jpg")

        if o_filter == 'Original (RGB)':
            with col4:
                st.image(path_resources + "filter_rgb.png",width=200)
            load_resources('Save_history_CNN_FS_60x60_rgb.npy','Classification_Report_CNN_FS_rgb.csv','Conf_matrix_CNN_FS_rgb.jpg')
            model_name = 'Save_model_CNN_FS_60x60_rgb.h5'

        elif o_filter =='Kmeans_1':
            with col4:
                st.image(path_resources + "filter_kmeans1.png",width=200)
            load_resources('Save_history_CNN_FS_60x60_kmeans1.npy','Classification_Report_CNN_FS_kmeans1.csv','Conf_matrix_CNN_FS_kmeans1.jpg')
            model_name = 'Save_model_CNN_FS_60x60_kmeans1.h5'

        elif o_filter == 'Kmeans_2':
            with col4:
                st.image(path_resources + "filter_kmeans2.png",width=200)
            load_resources('Save_history_CNN_FS_60x60_kmeans2.npy','Classification_Report_CNN_FS_kmeans2.csv','Conf_matrix_CNN_FS_kmeans2.jpg')
            model_name = 'Save_model_CNN_FS_60x60_kmeans2.h5'

        elif o_filter == 'Kmeans XYRGB':
            with col4:
                st.image(path_resources + "filter_kmeansxyrgb.png",width=200)
            load_resources('Save_history_CNN_FS_60x60_kmeansXYRGB.npy','Classification_Report_CNN_FS_kmeansXYRGB.csv','Conf_matrix_CNN_FS_kmeansXYRGB.jpg')
            model_name = 'Save_model_CNN_FS_60x60_kmeansXYRGB.h5'

        elif o_filter == 'Equalizer':
            with col4:
                st.image(path_resources + "filter_equalizer.png",width=200)
            load_resources('Save_history_CNN_FS_60x60_Equalizer.npy','Classification_Report_CNN_FS_Equalizer.csv','Conf_matrix_CNN_FS_equalizer.jpg')
            model_name = 'Save_model_CNN_FS_60x60_Equalizer.h5'

        elif o_filter == 'Mean Shift':
            with col4:
                st.image(path_resources + "filter_meanshift.png",width=200)
            load_resources('Save_history_CNN_FS_60x60_mean_shift.npy','Classification_Report_CNN_FS_mean_shift.csv','Conf_matrix_CNN_FS_mean_shift.jpg')
            model_name = 'Save_model_CNN_FS_60x60_mean_shift.h5'

    elif o_model_dl == 'VGG16' :
        with col2:
            st.image(path_resources + "model_vgg16.jpg")

        if o_filter == 'Original (RGB)':
            with col4:
                st.image(path_resources + "filter_rgb.png",width=200)
            load_resources('Save_history_CNN_VGG16_60x60_rgb.npy','Classification_Report_CNN_VGG16_rgb.csv','Conf_matrix_CNN_VGG16_rgb.jpg')
            model_name = 'Save_model_CNN_VGG16_60x60_rgb.h5'
     
        elif o_filter =='Kmeans_1':
            with col4:
                st.image(path_resources + "filter_kmeans1.png",width=200)
            load_resources('Save_history_CNN_VGG16_60x60_kmeans1.npy','Classification_Report_CNN_VGG16_kmeans1.csv','Conf_matrix_CNN_VGG16_kmeans1.jpg')
            model_name = 'Save_model_CNN_VGG16_60x60_kmeans1.h5'

        elif o_filter == 'Kmeans_2':
            with col4:
                st.image(path_resources + "filter_kmeans2.png",width=200)
            load_resources('Save_history_CNN_VGG16_60x60_kmeans2.npy','Classification_Report_CNN_VGG16_kmeans2.csv','Conf_matrix_CNN_VGG16_kmeans2.jpg')
            model_name = 'Save_model_CNN_VGG16_60x60_kmeans2.h5'

        elif o_filter == 'Kmeans XYRGB':
            with col4:
                st.image(path_resources + "filter_kmeansxyrgb.png",width=200)
            load_resources('Save_history_CNN_VGG16_60x60_kmeansXYRGB.npy','Classification_Report_CNN_VGG16_kmeansXYRGB.csv','Conf_matrix_CNN_VGG16_kmeansXYRGB.jpg')
            model_name = 'Save_model_CNN_VGG16_60x60_kmeansXYRGB.h5'

        elif o_filter == 'Equalizer':
            with col4:
                st.image(path_resources + "filter_equalizer.png",width=200)
            load_resources('Save_history_CNN_VGG16_60x60_Equalizer.npy','Classification_Report_CNN_VGG16_Equalizer.csv','Conf_matrix_CNN_VGG16_equalizer.jpg')
            model_name = 'Save_model_CNN_VGG16_60x60_Equalizer.h5'

        elif o_filter == 'Mean Shift':
            with col4:
                st.image(path_resources + "filter_meanshift.png",width=200)
            load_resources('Save_history_CNN_VGG16_60x60_mean_shift.npy','Classification_Report_CNN_VGG16_mean_shift.csv','Conf_matrix_CNN_VGG16_mean_shift.jpg')            
            model_name = 'Save_model_CNN_VGG16_60x60_mean_shift.h5'
 
    elif o_model_dl == 'VGG19' :
        with col2:
            st.image(path_resources + "model_vgg19.jpg")

        if o_filter == 'Original (RGB)':
            with col4:
                st.image(path_resources + "filter_rgb.png",width=200)
            load_resources('Save_history_CNN_VGG19_60x60_rgb.npy','Classification_Report_CNN_VGG19_rgb.csv','Conf_matrix_CNN_VGG19_rgb.jpg')
            model_name = 'Save_model_CNN_VGG19_60x60_rgb.h5'
       
        elif o_filter =='Kmeans_1':
            with col4:
                st.image(path_resources + "filter_kmeans1.png",width=200)
            load_resources('Save_history_CNN_VGG19_60x60_kmeans1.npy','Classification_Report_CNN_VGG19_kmeans1.csv','Conf_matrix_CNN_VGG19_kmeans1.jpg')
            model_name = 'Save_model_CNN_VGG19_60x60_kmeans1.h5'

        elif o_filter == 'Kmeans_2':
            with col4:
                st.image(path_resources + "filter_kmeans2.png",width=200)
            load_resources('Save_history_CNN_VGG19_60x60_kmeans2.npy','Classification_Report_CNN_VGG19_kmeans2.csv','Conf_matrix_CNN_VGG19_kmeans2.jpg')
            model_name = 'Save_model_CNN_VGG19_60x60_kmeans2.h5'

        elif o_filter == 'Kmeans XYRGB':
            with col4:
                st.image(path_resources + "filter_kmeansxyrgb.png",width=200)
            load_resources('Save_history_CNN_VGG19_60x60_kmeansXYRGB.npy','Classification_Report_CNN_VGG19_kmeansXYRGB.csv','Conf_matrix_CNN_VGG19_kmeansXYRGB.jpg')
            model_name = 'Save_model_CNN_VGG19_60x60_kmeansXYRGB.h5'

        elif o_filter == 'Equalizer':
            with col4:
                st.image(path_resources + "filter_equalizer.png",width=200)
            load_resources('Save_history_CNN_VGG19_60x60_Equalizer.npy','Classification_Report_CNN_VGG19_Equalizer.csv','Conf_matrix_CNN_VGG19_equalizer.jpg')
            model_name = 'Save_model_CNN_VGG19_60x60_Equalizer.h5'

        elif o_filter == 'Mean Shift':
            with col4:
                st.image(path_resources + "filter_meanshift.png",width=200)
            load_resources('Save_history_CNN_VGG19_60x60_mean_shift.npy','Classification_Report_CNN_VGG19_mean_shift.csv','Conf_matrix_CNN_VGG19_mean_shift.jpg')            
            model_name = 'Save_model_CNN_VGG19_60x60_mean_shift.h5'

    elif o_model_dl == 'ResNet50' :
        with col2:
            st.image(path_resources + "model_resnet50.jpg")

        if o_filter == 'Original (RGB)':
            with col4:
                st.image(path_resources + "filter_rgb.png",width=200)
            load_resources('Save_history_CNN_ResNet50_60x60_rgb.npy','Classification_Report_CNN_ResNet50_rgb.csv','Conf_matrix_CNN_ResNet50_rgb.jpg')
            model_name = 'Save_model_ResNet50_60x60_rgb.h5'
        
        elif o_filter =='Kmeans_1':
            with col4:
                st.image(path_resources + "filter_kmeans1.png",width=200)
            load_resources('Save_history_CNN_ResNet50_60x60_kmeans1.npy','Classification_Report_CNN_ResNet50_kmeans1.csv','Conf_matrix_CNN_ResNet50_kmeans1.jpg')
            model_name = 'Save_model_CNN_ResNet50_60x60_kmeans1.h5'

        elif o_filter == 'Kmeans_2':
            with col4:
                st.image(path_resources + "filter_kmeans2.png",width=200)
            load_resources('Save_history_CNN_ResNet50_60x60_kmeans2.npy','Classification_Report_CNN_ResNet50_kmeans2.csv','Conf_matrix_CNN_ResNet50_kmeans2.jpg')
            model_name = 'Save_model_CNN_ResNet50_60x60_kmeans2.h5'

        elif o_filter == 'Kmeans XYRGB':
            with col4:
                st.image(path_resources + "filter_kmeansxyrgb.png",width=200)
            load_resources('Save_history_CNN_ResNet50_60x60_kmeansXYRGB.npy','Classification_Report_CNN_ResNet50_kmeansXYRGB.csv','Conf_matrix_CNN_ResNet50_kmeansXYRGB.jpg')
            model_name = 'Save_model_CNN_ResNet50_60x60_kmeansXYRGB.h5'

        elif o_filter == 'Equalizer':
            with col4:
                st.image(path_resources + "filter_equalizer.png",width=200)
            load_resources('Save_history_CNN_ResNet50_60x60_Equalizer.npy','Classification_Report_CNN_ResNet50_Equalizer.csv','Conf_matrix_CNN_ResNet50_equalizer.jpg')
            model_name = 'Save_model_CNN_ResNet50_60x60_Equalizer.h5'

        elif o_filter == 'Mean Shift':
            with col4:
                st.image(path_resources + "filter_meanshift.png",width=200)
            load_resources('Save_history_CNN_ResNet50_60x60_mean_shift.npy','Classification_Report_CNN_ResNet50_mean_shift.csv','Conf_matrix_CNN_ResNet50_mean_shift.jpg')            
            model_name = 'Save_model_CNN_ResNet50_60x60_mean_shift.h5'

    elif o_model_dl == 'Xception' :
        with col2:
            st.image(path_resources + "model_xception.jpg")

        if o_filter == 'Original (RGB)':
            with col4:
                st.image(path_resources + "filter_rgb.png",width=200)
            load_resources('Save_history_CNN_Xception_60x60_rgb.npy','Classification_Report_CNN_Xception_rgb.csv','Conf_matrix_CNN_Xception_rgb.jpg')
            model_name = 'Save_model_CNN_Xception_60x60_rgb.h5'
       
        elif o_filter =='Kmeans_1':                        
            with col4:
                st.image(path_resources + "filter_kmeans1.png",width=200)
            load_resources('Save_history_CNN_Xception_60x60_kmeans1.npy','Classification_Report_CNN_Xception_kmeans1.csv','Conf_matrix_CNN_Xception_kmeans1.jpg')
            model_name = 'Save_model_CNN_Xception_60x60_kmeans1.h5'

        elif o_filter == 'Kmeans_2':
            with col4:
                st.image(path_resources + "filter_kmeans2.png",width=200)
            load_resources('Save_history_CNN_Xception_60x60_kmeans2.npy','Classification_Report_CNN_Xception_kmeans2.csv','Conf_matrix_CNN_Xception_kmeans2.jpg')
            model_name = 'Save_model_CNN_Xception_60x60_kmeans2.h5'

        elif o_filter == 'Kmeans XYRGB':
            with col4:
                st.image(path_resources + "filter_kmeansxyrgb.png",width=200)
            load_resources('Save_history_CNN_Xception_60x60_kmeansXYRGB.npy','Classification_Report_CNN_Xception_kmeansXYRGB.csv','Conf_matrix_CNN_Xception_kmeansXYRGB.jpg')
            model_name = 'Save_model_CNN_Xception_60x60_kmeansXYRGB.h5'

        elif o_filter == 'Equalizer':
            with col4:
                st.image(path_resources + "filter_equalizer.png",width=200)
            load_resources('Save_history_CNN_Xception_60x60_Equalizer.npy','Classification_Report_CNN_Xception_Equalizer.csv','Conf_matrix_CNN_Xception_equalizer.jpg')
            model_name = 'Save_model_CNN_Xception_60x60_Equalizer.h5'

        elif o_filter == 'Mean Shift':
            with col4:
                st.image(path_resources + "filter_meanshift.png",width=200)
            load_resources('Save_history_CNN_Xception_60x60_mean_shift.npy','Classification_Report_CNN_Xception_mean_shift.csv','Conf_matrix_CNN_Xception_mean_shift.jpg')          
            model_name = 'Save_model_CNN_Xception_60x60_mean_shift.h5'
    
#==== Prediction on a specific cell
#===============================================================================

    st.markdown("<h2 style='text-align: left; color: #00008B;'>Cell type prediction from our dataset</h2>", unsafe_allow_html=True)
    st.info("###### Now, let's try the selected model and filter on a random image and see if the type prediction is correct !")

    col_left, col_center, col_right = st.columns((0.5,0.05,0.5))

    with col_left:
        st.info ("#### Information\n\n"+ 
        "###### The image used and randomly selected, comes from a set of unclassified images\n\n" +
        "###### The resolution image is: 60x60\n\n" +
        "###### The selected model is: " + o_model_dl + "\n\n" +
        "###### The selected filter is: " + o_filter)

    if st.button('Click me to make a prediction !!!'):
        # Liste des adresses des images présentes dans le dossier data_sample
        files = [filename.replace('\\','/') for filename in glob.glob('data_sample/*/*.jpg')]
        
        # On choisit au hasard une adresse d'image
        filename = np.random.choice(files,1)[0]
            
        # On stocke le nom du dossier où est stockée l'image selectionnée au hasard
        folder_name_img = filename.split('/')[-2]

        # On stocke l'image en RGB
        img_rgb = cv2.imread(filename,cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        # Dans une liste, on stocke les transformations par des filtres de l'image 
        imgs_filtered = [img_rgb
                        ,filter_Kmeans1(filename)
                        ,filter_Kmeans2(filename)
                        ,filter_KmeansXYRGB(filename)
                        ,EqualizerImg(filename)
                        ,filter_MeanShift(filename)]

        # On redimensionne les images de la liste en 60x60
        imgs_filtered = [cv2.resize(img,dsize = (60,60)) for img in imgs_filtered]


        # On crée un dictionnaire associant le nom de chaque filtre à un indice
        filter2idx = {u:i for i, u in enumerate(option_filter)}

        # On stocke l'image de la liste correspondant au filtre selectionné par l'utilisateur à travers le dictionnaire
        img_f = imgs_filtered[filter2idx[o_filter]]

        # Pour la futur prédiction, on normalise la valeur des pixels de l'image
        img_tf = image.img_to_array(img_f)/255
        #On ajoute une dimension à l'image
        img_tf = np.expand_dims(img_tf, axis = 0)

        # On charge le modèle choisit par l'utilisateur
        model = tf.keras.models.load_model(path_models + model_name) #Chargement du modèle
        
        # On stocke les valeurs de probabilité des classes de l'image
        probas = model.predict(img_tf)[0]
        # On stocke la valeur de la probabilité la plus élevée
        pred_proba = np.max(probas)*100

        # On crée la liste des noms des différents dossiers de types de cellules présents dans le dossier data_samples
        cell_types = ['neutrophil', 'eosinophil', 'ig', 'platelet', 'erythroblast', 'monocyte','basophil','lymphocyte']
        # On crée la liste des noms des différentes types de cellules que l'on souhaite afficher à l'avenir
        cell_types2 = ['neutrophil', 'eosinophil', 'immature granulocyte', 'platelet', 'erythroblast', 'monocyte','basophil','lymphocyte']

        # On crée un dictionnaire associant le nom de chaque dossier de clatypessses de cellules à un indice
        ct2idx = {u:i for i,u in enumerate(cell_types)}

        # On stocke le nom du type correspondant à l'indice de la probabilité la plus élevé 
        pred_cell_type = cell_types2[np.argmax(probas)]
        # On stocke le nom du type de cellule de l'image choisi au hasard
        true_cell_type = cell_types2[ct2idx[folder_name_img]]

        if(true_cell_type ==pred_cell_type):
            st.success ("#### Prediction results\n\n" + 
            "###### The prediction is CORRECT !!\n\n" +
            "###### The true cell type is " + true_cell_type.upper() + "\n\n" +
            "###### The predicted cell type is: " + pred_cell_type.upper() + "\n\n" +
            "###### The model was sure to find the true type with probability of " + str(round(pred_proba,1)) + "%")
        else:
            st.error ("#### Prediction results\n\n" + 
            "###### The prediction is NOT CORRECT !!\n\n" +
            "###### The true cell type is " + true_cell_type.upper() + "\n\n" +
            "###### The predicted cell type is: " + pred_cell_type.upper() + "\n\n" +
            "###### Yet, the model was sure to find the true type with probability of " + str(round(pred_proba,1)) + "%")
        
        fig = plt.figure(figsize=(10, 10))
        with col_right:
            plt.subplot(1,2,1)
            plt.title("Original image", fontsize=20)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img_rgb)
            
            plt.subplot(1,2,2)
            plt.title("Analysed image", fontsize=20)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img_f)
            st.pyplot(fig)
            
#==== It's your turn
#===============================================================================
        
    ##############-------------------------- Partie Prédiction --------------------------##############
    temp_filename_wo_ext = "tmp_img/dropped_image_DL"

    st.markdown("<h2 style='text-align: left; color: #00008B;'>Cell type prediction with your image</h2>", unsafe_allow_html=True)
    # Uploader l'image à qui nous voulons faire une prédiction  
    st.info("###### To finish, upload your cell image and see the prediction of the selected model with the selected filter !!")

    col_left2, col_center, col_right2 = st.columns((0.5,0.05,0.5))
    with col_left2:
    
        st.info ("#### Information\n\n"+ 
                "###### The selected model is: " + o_model_dl + "\n\n" +
                "###### The selected filter is: " + o_filter)
        
        dropped_img = st.file_uploader("Upload ", type=["jpg","jpeg","png","tiff"])
        if dropped_img is not None: 
            img_p = Image.open(dropped_img)
            # ext = dropped_img.name.split('.')[-1]
            # temp_filename = temp_filename_wo_ext + '.' + ext
            temp_filename = "tmp_img/" + dropped_img.name
            img_p.save(temp_filename) # sauvegarde en fichier temporaire
            img_p = cv2.imread(temp_filename,cv2.IMREAD_COLOR)
            img_p = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)
            
    show_results = st.button('Click me to make new prediction !!!')
    if show_results : 
        imgs_filtered2 = [img_p
                    ,filter_Kmeans1(temp_filename)
                    ,filter_Kmeans2(temp_filename)
                    ,filter_KmeansXYRGB(temp_filename)
                    ,EqualizerImg(temp_filename)
                    ,filter_MeanShift(temp_filename)]

        imgs_filtered2 = [cv2.resize(i,dsize = (60,60)) for i in imgs_filtered2]

        # On crée un dictionnaire associant le nom de chaque filtre à un indice
        filter2idx = {u:i for i, u in enumerate(option_filter)}

        # On stocke l'image de la liste correspondant au filtre selectionné par l'utilisateur à travers le dictionnaire
        img_f2 = imgs_filtered2[filter2idx[o_filter]]

        # Pour la future prédiction, on normalise la valeur des pixels de l'image
        img_tf2 = image.img_to_array(img_f2)/255
        #On ajoute une dimension à l'image
        img_tf2 = np.expand_dims(img_tf2, axis = 0)

        # On charge le modèle choisit par l'utilisateur
        model2 = tf.keras.models.load_model(path_models + model_name) #Chargement du modèle
        
        # On stocke les valeurs de probabilité des classes de l'image
        probas2 = model2.predict(img_tf2)[0]
        # On stocke la valeur de la probabilité la plus élevée
        pred_proba2 = np.max(probas2)*100

        # On crée la liste des noms des différentes types de cellules que l'on souhaite afficher à l'avenir
        cell_types2 = ['neutrophil', 'eosinophil', 'immature granulocyte', 'platelet', 'erythroblast', 'monocyte','basophil','lymphocyte']

        # On stocke le nom du type correspondant à l'indice de la probabilité la plus élevé 
        pred_cell_type2 = cell_types2[np.argmax(probas2)]
        # On stocke le nom du type de cellule de l'image choisi au hasard
        
        if pred_cell_type2 == 'neutrophil' :
            cell_info = "In cell biology, a neutrophilic body is a cell body that has an affinity for neutral dyes: \
                    cations as well as anions of the preparation stain the cell and reveal its structures. This is \
                    the case, for example, of a type of immune cell: the **neutrophilic granulocytes**.."

        elif pred_cell_type2 == 'eosinophil' : 
            cell_info = "**Eosinophilic granulocytes** are the rarest white blood cells in the bloodstream. Their name comes\
                    from the protein-rich granules in their cytoplasm, but also from their affinity for eosin, which \
                    colors them red (visible under light microscopy). They are als  called eosinophilic polynuclei."

        elif pred_cell_type2 == 'immature granulocyte' :
            cell_info = "**Immature granulocytes** are immature white blood cells.The presence of immature granulocytes in \
                    blood test results usually means that your body is fighting an infection or inflammation."

        elif pred_cell_type2 == 'platelet' :
            cell_info = "**Platelets** are also called thrombocytes. They are made in the bone marrow and help the blood \
                    to clot."
        elif pred_cell_type2 == 'erythroblast' :
            cell_info = "**Erythroblasts** are young red blood cells, which are made in the bone marrow. They lose their \
                    nucleus, and gain hemoglobin as they grow to become mature red blood cells."

        elif pred_cell_type2 == 'monocyte' :
            cell_info = "**Monocytes** are large mobile blood cells (20 to 40 micrometers in diameter) produced by the \
                    bone marrow from hematopoietic cells, and more specifically from monoblasts."
            
        elif pred_cell_type2 == 'basophil' :
            cell_info = "**Basophiles** are white blood cells whose nuclei contain granules. \
                Their number increases during bone marrow diseases and decreases during severe allergic reactions."
                    
        elif pred_cell_type2 == 'lymphocyte' :
            cell_info = "The **lymphocytes** are a variety of white blood cells. They are part of the leukocyte (white\
                    blood cell) family. They are small, round cells (about 7 microns in diameter) with a nucleus. \
                    They are found in the blood, bone marrow (where they are produced) and lymphoid tissue \
                    (spleen, lymph nodes)."
            



        st.info ("#### Prediction results\n\n" + 
        "###### The predicted cell type is: " + pred_cell_type2.upper() + "\n\n" +
        "###### The model was sure to find the true type with probability of " + str(round(pred_proba2,1)) + "%"+ "\n\n" + cell_info)
        
        fig2 = plt.figure(figsize=(10, 10))
        with col_right2:
            plt.subplot(1,2,1)
            plt.title("Dropped image\n"+dropped_img.name, fontsize=14)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img_p)
            
            plt.subplot(1,2,2)
            plt.title("Analysed image", fontsize=14)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img_f2)
            st.pyplot(fig2)
        
                

            

     

    
    