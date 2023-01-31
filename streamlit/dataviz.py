import streamlit as st
import pandas as pd
import numpy as np
#from termcolor import colored
import random
import inspect

#import sys
import glob
import os

import cv2 ## Package pour lire nos images
#from PIL import Image ## Importer ce package pour pouvoir voir le format de l'image
#import skimage.exposure

import matplotlib
#import seaborn as sns
import matplotlib.pyplot as plt

from Filtering import filter_Kmeans1,filter_Kmeans2,filter_KmeansXYRGB,filter_MeanShift,EqualizerImg

import warnings
warnings.filterwarnings("ignore")
import streamlit.components.v1 as components


#####--------------------------LISTES-----------------------------------#####

# Listes
noms_classe = ['basophil', 'eosinophil', 'erythroblast', 'immature granulocyte'
                , 'lymphocyte', 'monocyte','neutrophil','platelet']
#folder_names = os.listdir(path)
list_filters = ['RGB','Kmeans_1','Kmeans_2','KmeansXYRGB','Equalizer','Mean_shift']
list_filter_names = ['Original (RGB)','Kmeans(6+2)','Th + Kmeans3','Kmeans XYRGB','Equalizer','Mean Shift']
#option_resolution = ['360 x 360','180 x 180','90 x 90','60 x 60','45 x 45','30 x 30']

# Répertoire des images
path = 'data_sample/'

#####--------------------------FUNCTION DEFINITIONS-----------------------------------#####

# Mise en page des textes en markdown
def text2mkd(text):
    text = '>' + text
    text = text.replace('\n','\n>')
    return text

# Fonction pour récupérer k images de manière aléatoire

@st.cache(suppress_st_warning=True)
def RandomImage(k, noms_classes):
    '''
    la fonction RandomImage va nous permettre de récupérer un nombre d'images (k) de chaque classe de nos cellules
    de manière aléatoire, ensuite les stocker dans une liste img_random

    paramètres :
    noms_classes : les noms de nos 8 classes de cellules sanguines
    k : le nombre d'images qu'on veut récupérer de chaques classe de notre data base
    '''
    img_random = []
    for i in range(len(noms_classes)):
        for j in range(k):
            img = random.choice(noms_classes[i])
            img_random.append(img)
    return img_random

@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def viz_image(img):
    fig = plt.figure()
    plt.imshow(img)
    #plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    return fig


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def viz_filters(filename,resolution):

    img = cv2.imread(filename,cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_filtered = [img_rgb
                    ,filter_Kmeans1(filename)
                    ,filter_Kmeans2(filename)
                    ,filter_KmeansXYRGB(filename)
                    ,EqualizerImg(filename)
                    ,filter_MeanShift(filename)]

    list_filter_names

    fig = plt.figure(figsize=(13, 12))
    #c2.subheader("All Filters")
    
    k=1
    for img, f in zip(img_filtered,list_filter_names):
        
        img = cv2.resize(img,dsize = (resolution,resolution))
        
        # subplots
        plt.subplot(4, 4, k)
        #plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
        plt.gca().set_title(f, size=14) # title
        
        k+=2 if k ==3 else 1

    return fig


@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def viz_cell_types(filenames,o_filter,resolution):

    fig = plt.figure(figsize=(13, 12))
    
    for i in range(len(filenames)):
        
        # subplots
        plt.subplot(4, 4, i + 1)
        #plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        # set the title to subplots
        plt.gca().set_title(noms_classe[i], size=14)
        
        filename = path + os.listdir(path)[i] + '/' + filenames[i]
                     
        if o_filter == 'RGB':
            img = cv2.imread(filename,cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        elif o_filter == 'Kmeans_1':
            img = filter_Kmeans1(filename)
        
        elif o_filter == 'Kmeans_2':
            img = filter_Kmeans2(filename)
            
        elif o_filter == 'KmeansXYRGB':
            img = filter_KmeansXYRGB(filename)
        
        elif o_filter == 'Equalizer' :
            img = EqualizerImg(filename)
            
        elif o_filter == 'Mean_shift':
           img = filter_MeanShift(filename)
           
        img = cv2.resize(img,dsize = (resolution,resolution))
        
        plt.imshow(img)
    
    return fig
        

@st.cache(suppress_st_warning=True)
def filter_img(filename, resolution=None):
    
    img = cv2.imread(filename,cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_filtered = [img_rgb
                    ,filter_Kmeans1(filename)
                    ,filter_Kmeans2(filename)
                    ,filter_KmeansXYRGB(filename)
                    ,EqualizerImg(filename)
                    ,filter_MeanShift(filename)]
    
    if resolution != None:
        img_filtered = [cv2.resize(img,dsize = (resolution,resolution)) for img in img_filtered]
    
    return img_filtered

#===============================================================================
#==== 3.1 Data
#===============================================================================
    
    
## Créer une liste où stocker les noms de nos 8 différentes classes
Classes = []
for element in os.listdir(path):
    nom_classe = os.listdir(path + element)
    Classes.append(nom_classe)

# importer toutes les images des 8 différentes classes et les stocker dans un array images
Images = []
for element in os.listdir(path):
    for img in glob.glob(path + element + ' /' + '*.jpg'):
        imge = cv2.imread(img)
        Images.append(imge)
        
img_random = RandomImage(1, Classes)


#####-----------------------------PAGE DATAVIZ----------------------------#####

def dataviz():
    # st.title("DataViz")

    st.markdown("<h1 style='text-align: left; color: #A52A2A;'>DataViz</h1>", unsafe_allow_html=True)
    
    
    #========== Read dataset =======================================================

    img_info = pd.read_csv('resources/img_info.csv')
    cell_types = pd.read_csv('resources/cell_types.csv')
    # cell_types = cell_types[cell_types.Source=='article'].drop(columns=['Source'])
    df_imgs = pd.merge(img_info,cell_types,how='outer') # merge on column cell_subtype
    # print(len(df_imgs),len(img_info))
    # print('\n',df_imgs.cell_type2.value_counts(),'\n')

    df2 = df_imgs.groupby(['cell_type2','cell_type_code','cell_subtype','cell_subtype_key']).agg({'filename':'count'}).reset_index()
    df2 = df2.rename(columns = {'cell_type2':'cell type'
                                ,'cell_type_code':'cell type short'
                                ,'cell_subtype':'cell subtype'
                                ,'cell_subtype_key':'cell subtype short'
                                ,'filename':'number of cells'})

    df3 = df_imgs.groupby(['cell_type2','cell_type_code']).agg({'filename':'count'}).reset_index()
    df3 = df2.rename(columns = {'cell_type2':'cell type'
                                ,'cell_type_code':'cell type short'
                                ,'filename':'number of cells'})

    df_dim = df_imgs.groupby(['img_dim']).agg({'filename':'count'}).reset_index()\
             .rename(columns = {'img_dim':'image dimension'
                                ,'filename':'number of files'})

    # st.dataframe(df_imgs)
    # st.dataframe(df2)
    # st.dataframe(df3)
    # st.dataframe(df_dim)
    
    #===============================================================================
    #==== Distribution of cell types
    #===============================================================================
    # st.markdown('## **Our Dataset**') # markdown
    # st.header('Presentation of the dataset')
    st.markdown("<h2 style='text-align: left; color: #00008B;'>Presentation of the dataset</h2>", unsafe_allow_html=True)
    col1,col2 = st.columns(2)

    #===============================================================================
    # Plotly
    #===============================================================================

    prez = pd.read_excel(open('prez.xlsx', 'rb'),sheet_name='dataviz', index_col=0)
    intro_dataset_text = text2mkd(prez.loc['intro_dataset','text'])
    col1.markdown(intro_dataset_text)

    import plotly.express as px


    # Pie
    fig = px.pie(df3
                , values='number of cells'
                ,hover_data=['cell type']
                , names='cell type'
                #  , names='cell type short'
                # ,title="Distribution of blood cell type in the dataset"
                 , color_discrete_sequence=px.colors.sequential.Cividis #Burg RdBu algae Blues Cividis
                 , hole=.3
                 )
    fig.update_traces(
        textposition='inside'
        , textinfo='percent+label'
        , textfont_size=15
        ,marker=dict(line=dict(color='white', width=4))
        # ,font=dict(size=15)
        )
    fig.update_layout(
        margin=dict(l=0,r=0,b=0,t=0,pad=4)
        # ,paper_bgcolor="LightSteelBlue"
        ,legend_title="Types of blood cells"
        ,font=dict(size=15)
        ,legend = dict(font = dict(size = 17, color = "black"))
    )

    col2.plotly_chart(fig)

    col1.markdown("""</br>""", unsafe_allow_html=True)
    col1.markdown("""</br>""", unsafe_allow_html=True)
    col1.markdown("""</br>""", unsafe_allow_html=True)

    url1 = "https://data.mendeley.com/datasets/snkd93bnjr/1"
    col1.markdown('**Access to dataset** [link](%s)' % url1, unsafe_allow_html=True)
    url2 = "https://www.sciencedirect.com/science/article/abs/pii/S0169260719303578?via%3Dihub"
    col1.markdown('**Access to article** [link](%s)' % url2, unsafe_allow_html=True)

    #===============================================================================
    #==== 2 Segmentation / filters presentation
    #===============================================================================

    st.markdown("""</br>""", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; color: #00008B;'>Image filtering & segmentation</h2>", unsafe_allow_html=True)

    prez = pd.read_excel(open('prez.xlsx', 'rb'),sheet_name='dataviz', index_col=0)
    intro_filters_text = text2mkd(prez.loc['intro_filters','text'])
    st.markdown(intro_filters_text)

    # ouvrir le fichier xlsx avec les parties texte
    prez = pd.read_excel(open('prez.xlsx', 'rb'),sheet_name='filters')

    # viz_filters('data_sample/erythroblast/ERB_703985.jpg',prez.filter_name,prez.filter_description_en)

    # ouvrir une image de cellule et lui appliquer les différents filtres
    images = filter_img('data_sample/erythroblast/ERB_703985.jpg')
    
    for img,filter_name,filter_desc in zip(images,prez.filter_name,prez.filter_description_en):
        
        col1,col2 = st.columns([1, 5]) 
        
        # option 1 : calculation of filters is done on the fly:
        fig = viz_image(img)
        col1.pyplot(fig)
        
        # Option 2: pre-saved files (issue : the size of the image is not controlled)
        #image = 'data_sample/erythroblast/ERB_703985.jpg'
        #col1.image(image, caption=None, width=700, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

        col2.markdown('#### '+ filter_name)
        col2.markdown(text2mkd(filter_desc))

    

    #===============================================================================
    #==== 3 INTERACTION
    #===============================================================================
    
    # #Header
    # html_temp = """
    # <div style="background-color:tomato;padding:1px">
    # <h3 style="color:white;text-align:center;"> Preprocessing of Images</h3>
    # </div><br>"""
    # st.markdown(html_temp, unsafe_allow_html=True)
    # st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)
    
    # Segmentation des classes - choix de filtre et resolution et appliquer a une image de chaque classe
    # st.header("Filter Visualisation interface : try yourself !")
    st.markdown("""</br>""", unsafe_allow_html=True)
    st.markdown("""</br>""", unsafe_allow_html=True)
    st.markdown("""</br>""", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; color: #00008B;'>Filter Visualisation interface : try yourself !</h2>", unsafe_allow_html=True)
    

    #===============================================================================
    #==== 3.2 Preprocessing type cellule sur tous les filtres
    #===============================================================================
    
    st.info('#### ...either by choosing your filter')
    
    c1, c2 = st.columns([2, 4])

    #c1.subheader('cell type Selection')
    c1.markdown('##### Cell type')
    o_class = c1.radio('',noms_classe, key = 'class_list')
    st.write(" ")
    #c1.subheader('Resolution Selection')
    c1.markdown('##### Resolution')
    # o_res = c1.slider('', 30, 360, 90)
    o_res1 = c1.slider('Resolution', min_value=30, max_value=360, step=30, value = 360, key ='resolution_filters')
    
    for i,n in enumerate(noms_classe):
        if n==o_class:
            p=i
    
    filename = path + os.listdir(path)[p] + '/' + img_random[p]

    fig = viz_filters(filename,o_res1)
    c2.markdown('##### All Filters applied to a single image')
    c2.pyplot(fig)
    
    #===============================================================================
    #==== 3.3 Preprocessing filtre sur tous les types de cellules
    #===============================================================================
    
    st.info('#### ...or by changing the cell type')
    
    c1, c2 = st.columns([2, 4])
    
    c1.markdown('##### Filter')
    o_filter = c1.radio(' ',list_filters, key = 'filter_list')
    st.write(" ")
    #c1.write(o_filter)

    c1.markdown('##### Resolution')
    # o_res = c1.slider('Resolution', 30, 360, 90)
    o_res = c1.slider('Resolution', min_value=30, max_value=360, step=30, value = 360, key ='resolution_cell_types')

    # ------- Afficher pour chaque type de cellules une image, qu'on récupére du manière aléatoire ---------#
    # selectionner une seule cellule de chaque type de classe et les visualiser avec plt.imshow
    
    
    c2.markdown('##### Filter applied on all blood cell types')

    fig = viz_cell_types(img_random,o_filter,o_res)
        
    c2.pyplot(fig)
