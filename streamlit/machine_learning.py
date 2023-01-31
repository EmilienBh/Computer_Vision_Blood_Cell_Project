import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
# from plotly.subplots import make_subplots
import glob
import cv2
import matplotlib.pyplot as plt
import joblib



from Filtering import filter_Kmeans1, filter_Kmeans2, filter_KmeansXYRGB, EqualizerImg, filter_MeanShift

path_resources = "resources/"
path_models =  "models/"


def load_resources(report_name, matrix_name):

    r_loaded = pd.read_csv(path_resources + report_name,index_col=0).astype(str).replace("nan", "") # Chargement du rapport
    r_loaded.index = ['neutrophil', 'eosinophil', 'ig','platelet','erythroblast','monocyte','basophil','lymphocyte','accuracy','macro avg','weighted avg']

    
    col_left, col_right = st.columns((1, 1))
    with col_right: 
        st.markdown("<h5 style='text-align: center; color: black;'>Confusion matrix</h5>", unsafe_allow_html=True)

        st.image(path_resources + matrix_name,width= 450) # Affichage de la matrice de confusion
    with col_left:
        st.markdown("<h5 style='text-align: center; color: black;'>Classification report</h5>", unsafe_allow_html=True)
        st.dataframe(r_loaded, height=340) # Affichage du rapport de classification
    
    
    
def Fonction_ML():

    st.markdown("<h1 style='text-align: left; color: #A52A2A;'>Machine Learning</h1>", unsafe_allow_html=True)

    st.markdown("""
                To be able to answer our problem of classification of the 8 types of blood cells, we initially looked \
                at two Machine learning models that have proven their performances in the past on images classification \
                problems, this models are :""") 

    st.markdown("""
                - **Random Forest** 
                - **SVM**
                        """)   
        
    st.markdown("""
                  We then did some fine tuning on Random Forest parameters, the model is named here by :
                      """)
    st.markdown(""" 
                - **RandomForestCV**
                """)
    st.markdown("""            
                  We did not do fine tuning on the SVM parameters because the latter requires more powerful hardware 
                  capabilities than those we have.  : 
                 
                    """)
                         
    st.markdown("""
                    For all our tests we split the datasets like this : **80%** for the training of the model and **20%** 
                    for the validation. 
                    """ )
    
    st.markdown("<h2 style='text-align: left; color: #00008B;'>Score results</h2>", unsafe_allow_html=True)
    st.info("###### In this part, you can see the results of your selections. These graphs below show the performance of the model in terms of the filter selected")
    
    st.markdown("<h2 style='text-align: left; color: #00008B;'>Model and filter selection</h2>", unsafe_allow_html=True)
    st.info("###### First, select a Machine learning model ")
    
    option_model = ['RandomForest','SVM','RandomForestCV']
    o_model = st.radio('Select model',option_model)
    
    st.info("###### Then, choose a filter image and an overview of the filter is displayed next to the filter selection.")
    col1, col2 = st.columns((0.3,1))

    
    with col1:
        option_fil = ['RGB','Kmeans_1','Kmeans_2','KmeansXYRGB','Equalizer','Mean_shift']
        o_filter = st.radio("Select filter",option_fil)
    
    if o_model == 'RandomForest' :
        if o_filter == 'RGB':
            with col2:
                st.image(path_resources + "filter_rgb.png",width=200)
            load_resources('Classification_Report_ML_RandomForest_RGB.csv','Conf_matrix_ML_RandomForest_RGB.jpg')
            model_name = 'RandomForest_rgb_model.sav'

        elif o_filter =='Kmeans_1':
            with col2:
                st.image(path_resources + "filter_kmeans1.png",width=200)
            load_resources('Classification_Report_ML_RandomForest_Kmeans_1.csv','Conf_matrix_ML_RandomForest_Kmeans_1.jpg')
            model_name = 'RandomForest_Kmeans_1_model.sav'

        elif o_filter == 'Kmeans_2':
            with col2:
                st.image(path_resources + "filter_kmeans2.png",width=200)
            load_resources('Classification_Report_ML_RandomForest_Kmeans_2.csv','Conf_matrix_ML_RandomForest_Kmeans_2.jpg')
            model_name = 'RandomForest_Kmeans_2_model.sav'

        elif o_filter == 'KmeansXYRGB':
            with col2:
                st.image(path_resources + "filter_kmeansxyrgb.png",width=200)
            load_resources('Classification_Report_ML_RandomForest_KmeansXYRGB.csv','Conf_matrix_ML_RandomForest_KmeansXYRGB.jpg')
            model_name = 'RandomForest_KmeansXYRGB_model.sav'

        elif o_filter == 'Equalizer':
            with col2:
                st.image(path_resources + "filter_equalizer.png",width=200)
            load_resources('Classification_Report_ML_RandomForest_Equalizer.csv','Conf_matrix_ML_RandomForest_Equalizer.jpg')
            model_name = 'RandomForest_Equalizer_model.sav'

        elif o_filter == 'Mean_shift':
            with col2:
                st.image(path_resources + "filter_meanshift.png",width=200)
            load_resources('Classification_Report_ML_RandomForest_Mean_shift.csv','Conf_matrix_ML_RandomForest_Mean_shift.jpg')
            model_name = 'RandomForest_Mean_shift_model.sav'

    elif o_model == 'SVM' :

        if o_filter == 'RGB':
            with col2:
                st.image(path_resources + "filter_rgb.png",width=200)
            load_resources('Classification_Report_ML_SVM_RGB.csv','Conf_matrix_ML_SVM_RGB.jpg')
            model_name = 'SVM_RGB_model.sav'
     
        elif o_filter =='Kmeans_1':
            with col2:
                st.image(path_resources + "filter_kmeans1.png",width=200)
            load_resources('Classification_Report_ML_SVM_Kmeans_1.csv','Conf_matrix_ML_SVM_Kmeans_1.jpg')
            model_name = 'SVM_Kmeans_1_model.sav'

        elif o_filter == 'Kmeans_2':
            with col2:
                st.image(path_resources + "filter_kmeans2.png",width=200)
            load_resources('Classification_Report_ML_SVM_Kmeans_2.csv','Conf_matrix_ML_SVM_Kmeans_2.jpg')
            model_name = 'SVM_Kmeans_2_model.sav'

        elif o_filter == 'KmeansXYRGB':
            with col2:
                st.image(path_resources + "filter_kmeansxyrgb.png",width=200)
            load_resources('Classification_Report_ML_SVM_KmeansXYRGB.csv','Conf_matrix_ML_SVM_KmeansXYRGB.jpg')
            model_name = 'SVM_KmeansXYRGB_model.sav'

        elif o_filter == 'Equalizer':
            with col2:
                st.image(path_resources + "filter_equalizer.png",width=200)
            load_resources('Classification_Report_ML_SVM_Equalizer.csv','Conf_matrix_ML_SVM_Equalizer.jpg')
            model_name = 'SVM_KmeansXYRGB_model.sav'

        elif o_filter == 'Mean_shift':
            with col2:
                st.image(path_resources + "filter_meanshift.png",width=200)
            load_resources('Classification_Report_ML_SVM_Mean_shift.csv','Conf_matrix_ML_SVM_Mean_shift.jpg')
            model_name = 'SVM_Mean_shift_model.sav'
 
    elif o_model == 'RandomForestCV' :
        if o_filter == 'RGB':
            with col2:
                st.image(path_resources + "filter_rgb.png",width=200)
            load_resources('Classification_Report_ML_RandomForestCV_RGB.csv','Conf_matrix_ML_RandomForestCV_RGB.jpg')
            model_name = 'RandomForestCV_RGB_model.sav'
       
        elif o_filter =='Kmeans_1':
            with col2:
                st.image(path_resources + "filter_kmeans1.png",width=200)
            load_resources('Classification_Report_ML_RandomForestCV_Kmeans_1.csv','Conf_matrix_ML_RandomForestCV_Kmeans_1.jpg')
            model_name = 'RandomForestCV_Kmeans_1_model.sav'

        elif o_filter == 'Kmeans_2':
            with col2:
                st.image(path_resources + "filter_kmeans2.png",width=200)
            load_resources('Classification_Report_ML_RandomForestCV_Kmeans_2.csv','Conf_matrix_ML_RandomForestCV_Kmeans_2.jpg')
            model_name = 'RandomForestCV_Kmeans_2_model.sav'

        elif o_filter == 'KmeansXYRGB':
            with col2:
                st.image(path_resources + "filter_kmeansxyrgb.png",width=200)
            load_resources('Classification_Report_ML_RandomForestCV_KmeansXYRGB.csv','Conf_matrix_ML_RandomForestCV_KmeansXYRGB.jpg')
            model_name = 'RandomForestCV_KmeansXYRGB_model.sav'

        elif o_filter == 'Equalizer':
            with col2:
                st.image(path_resources + "filter_equalizer.png",width=200)
            load_resources('Classification_Report_ML_RandomForestCV_Equalizer.csv','Conf_matrix_ML_RandomForestCV_Equalizer.jpg')
            model_name = 'RandomForestCV_Equalizer_model.sav'

        elif o_filter == 'Mean_shift':
            with col2:
                st.image(path_resources + "filter_meanshift.png",width=200)
            load_resources('Classification_Report_ML_RandomForestCV_Mean_shift.csv','Conf_matrix_ML_RandomForestCV_Mean_shift.jpg')
            model_name = 'RandomForestCV_Mean_shift_model.sav'

    
    
    st.markdown("<h2 style='text-align: left; color: #00008B;'>Cell type prediction</h2>", unsafe_allow_html=True)
    st.info("###### To finish, upload your cell image and let see the prediction of the selected model with the selected filter !!")

    # Uploader l'image à qui nous voulons faire une prédiction
    def resize_image(img):
        #img = cv2.imread(image_file)
        img = cv2.resize(img, (60,60), interpolation = cv2.INTER_AREA)
        return img
    
    col_left, col_center, col_right = st.columns((0.4,0.3,0.3))

    # temporary file name
    temp_filename_wo_ext = "tmp_img/dropped_image_ML"

    with col_left :
        st.info ("#### Information\n\n"+ 
            "###### The selected model is: " + o_model + "\n\n" +
            "###### The selected filter is: " + o_filter) 
        
        image_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png","tiff"])
        # save image in as a temporary file
        if image_file is not None: 
            img_p = Image.open(image_file)
            # ext = str(image_file.name).split('.')[-1]
            # temp_filename = temp_filename_wo_ext + '.' + ext
            temp_filename = "tmp_img/" + image_file.name
            img_p.save(temp_filename) # sauvegarde en fichier temporaire
    
    with col_center :
        if image_file is None: 
        #with col_left :
            img = Image.open('resources/digit.png')
            st.image(img, caption='')
        else :
            # img = Image.open(image_file)
            # st.image(img, caption='original Image ')
            img_p = cv2.imread(temp_filename,cv2.IMREAD_COLOR)
            img_p = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)

            fig = plt.figure(figsize=(10, 10))
            plt.title("Dropped image\n"+image_file.name, fontsize=14)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img_p)
            st.pyplot(fig)
    
    with col_right :    
        if image_file is None:
            img = Image.open('resources/digit.png')
            st.image(img, caption='')

        else:
        
            if o_filter == 'RGB' :

                # img = Image.open(image_file)
                # st.image(img, caption='original Image')
                img = cv2.imread(temp_filename,cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                graph_title = "original Image\n"+ o_filter

            elif o_filter == 'Kmeans_1' : 

                img = filter_Kmeans1(temp_filename) 
                # st.image(img, caption='Filtered Image\n'+ o_filter)
                graph_title = "Filtered Image\n"+ o_filter
        
            elif o_filter == 'Kmeans_2' :

                img = filter_Kmeans2(temp_filename)
                # st.image(img, caption='Filtered Image\n'+ o_filter)
                graph_title = "Filtered Image\n"+ o_filter
            
            elif o_filter == 'KmeansXYRGB' :

                img = filter_KmeansXYRGB(temp_filename)
                # st.image(img, caption='Filtered Image\n'+ o_filter)
                graph_title = "Filtered Image\n"+ o_filter
            
            elif o_filter == 'Equalizer' :

                img = EqualizerImg(temp_filename)
                # st.image(img, caption='Filtered Image\n'+ o_filter)
                graph_title = "Filtered Image\n"+ o_filter
            
            elif o_filter == 'Mean_shift' :

                img = filter_MeanShift(temp_filename)
                # st.image(img, caption='Filtered Image\n'+ o_filter)
                graph_title = 'Filtered Image\n'+ o_filter


            fig = plt.figure(figsize=(10, 10))
            plt.title(graph_title, fontsize=14)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
            st.pyplot(fig)


    # if o_model == 'RandomForest' or o_model == 'SVM' or o_model == 'RandomForestCV':
    #     if o_filter == 'RGB' or o_filter == 'Kmeans_1' or o_filter=='Kmeans_2' or o_filter== 'KmeansXYRGB' or o_filter=='Equalizer' or o_filter=='Mean_shift' :  
    #         load_image()
            
    show_results = st.button('Click me to make new prediction !!')
    if show_results : 

        # chargement du modèle
        classifier = joblib.load(path_models + o_model + '_' + o_filter+ '_model.sav')

        # st.write('Prédiction...')
        img = np.array(img)
        img = resize_image(img)
        prediction = classifier.predict(img.flatten().reshape(1,-1))

        # if prediction[0] == 'monocyte' :
        #     cell_info = """
        #                 **Monocytes** are large mobile blood cells (20 to 40 micrometers in diameter) produced by the 
        #                 bone marrow from hematopoietic cells, and more specifically from monoblasts.
        #                 """
                        
        # if prediction[0] == 'neutrophil' :
        #     cell_info = """
        #                 In cell biology, a neutrophilic body is a cell body that has an affinity for neutral dyes: 
        #                 cations as well as anions of the preparation stain the cell and reveal its structures. This is 
        #                 the case, for example, of a type of immune cell: the **neutrophilic granulocytes**..
        #                 """
        
        # if prediction[0] == 'eosinophil' : 
        #     cell_info = """
        #                 **Eosinophilic granulocytes** are the rarest white blood cells in the bloodstream. Their name comes
        #                 from the protein-rich granules in their cytoplasm, but also from their affinity for eosin, which 
        #                 colors them red (visible under light microscopy). They are als  called eosinophilic polynuclei.
        #                 """ 
            
        # if prediction[0] == 'basophil' :
        #     cell_info = """
        #                 **Basophiles** are white blood cells whose nuclei contain granules.  Their number increases during
        #                 bone marrow diseases and decreases during severe allergic reactions.
        #                 """
                        
        # if prediction[0] == 'lymphocite' :
        #     cell_info = """
        #                 The **lymphocytes** are a variety of white blood cells. They are part of the leukocyte (white
        #                 blood cell) family. They are small, round cells (about 7 microns in diameter) with a nucleus. 
        #                 They are found in the blood, bone marrow (where they are produced) and lymphoid tissue 
        #                 (spleen, lymph nodes).
        #                 """
            
        # if prediction[0] == 'erythroblast' :
        #     cell_info = """
        #                 **Erythroblasts** are young red blood cells, which are made in the bone marrow. They lose their 
        #                 nucleus, and gain hemoglobin as they grow to become mature red blood cells.
        #                 """

        # if prediction[0] == 'immature granulocytes' :
        #     cell_info = """
        #                 **Immature granulocytes** are immature white blood cells.The presence of immature granulocytes in 
        #                 blood test results usually means that your body is fighting an infection or inflammation.
        #                 """
                        
        # if prediction[0] == 'platelet' :
        #     cell_info = """
        #                 **Platelets** are also called thrombocytes. They are made in the bone marrow and help the blood 
        #                 to clot.
        #                 """

        # st.write(prediction[0])
        if prediction[0] == 'neutrophil' :
            cell_info = "In cell biology, a neutrophilic body is a cell body that has an affinity for neutral dyes: \
                    cations as well as anions of the preparation stain the cell and reveal its structures. This is \
                    the case, for example, of a type of immune cell: the **neutrophilic granulocytes**.."

        elif prediction[0] == 'eosinophil' : 
            cell_info = "**Eosinophilic granulocytes** are the rarest white blood cells in the bloodstream. Their name comes\
                    from the protein-rich granules in their cytoplasm, but also from their affinity for eosin, which \
                    colors them red (visible under light microscopy). They are als  called eosinophilic polynuclei."

        elif (prediction[0] == 'ig') or (prediction[0] == 'immature granulocytes'):
            cell_info = "**Immature granulocytes** are immature white blood cells.The presence of immature granulocytes in \
                    blood test results usually means that your body is fighting an infection or inflammation."

        elif prediction[0] == 'platelet' :
            cell_info = "**Platelets** are also called thrombocytes. They are made in the bone marrow and help the blood \
                    to clot."
        elif prediction[0] == 'erythroblast' :
            cell_info = "**Erythroblasts** are young red blood cells, which are made in the bone marrow. They lose their \
                    nucleus, and gain hemoglobin as they grow to become mature red blood cells."

        elif prediction[0] == 'monocyte' :
            cell_info = "**Monocytes** are large mobile blood cells (20 to 40 micrometers in diameter) produced by the \
                    bone marrow from hematopoietic cells, and more specifically from monoblasts."
            
        elif prediction[0] == 'basophil' :
            cell_info = "**Basophiles** are white blood cells whose nuclei contain granules. \
                Their number increases during bone marrow diseases and decreases during severe allergic reactions."
                    
        elif prediction[0] == 'lymphocyte' :
            cell_info = "The **lymphocytes** are a variety of white blood cells. They are part of the leukocyte (white\
                    blood cell) family. They are small, round cells (about 7 microns in diameter) with a nucleus. \
                    They are found in the blood, bone marrow (where they are produced) and lymphoid tissue \
                    (spleen, lymph nodes)."
        
        st.info('The Cell type predicted is : {}\n\n{}'.format(prediction[0].upper(),cell_info))

            
   


#Fonction_ML()