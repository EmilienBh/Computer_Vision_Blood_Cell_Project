import streamlit as st

st.set_page_config(
    page_title="Bloody Spy Blast",
    page_icon="resources/icon_b.png",
    # page_icon="resources/icon_w.png",
    layout="wide",
    initial_sidebar_state="auto" # "expanded" collapsed
 )

from intro import intro
from dataviz import dataviz
from machine_learning import Fonction_ML
from deep_learning import ma_fonction_dl
from conclusion import conclusion

import warnings
warnings.filterwarnings("ignore")

#####--------------------------SIDE BAR-----------------------------------#####

#Largeur de la sidebar
st.sidebar.markdown(f'''
    <style>
        section[data-testid="stSidebar"] .css-ng1t4o {{width: 30rem;}}
        section[data-testid="stSidebar"] .css-1d391kg {{width: 30rem;}}
    </style>
            ''', unsafe_allow_html=True)

#logo datascientest
st.sidebar.image("resources/datascientest.png",  caption=None, width=200, use_column_width=None,  output_format="auto", clamp=True, channels="RGB")
st.sidebar.markdown(" ")
st.sidebar.markdown(" ")
st.sidebar.markdown(" ")
st.sidebar.markdown(" ")

#st.markdown('<style>h1{color: #A52A2A;font-size: 70px}</style> ', unsafe_allow_html=True) # définit le style de tous les titres de la page
#st.markdown('<style>h2{color: #00008B;font-size: 60px}</style> ', unsafe_allow_html=True) # définit le style de tous les sous-titres de la page

#Affichage du Menu
page_names = ["Introduction", "DataViz", "Machine Learning", "Deep Learning", "Conclusion"]
sidebar = st.sidebar.radio("Menu", page_names)
st.sidebar.markdown(" ")
st.sidebar.markdown(" ")
st.sidebar.markdown(" ")

st.sidebar.markdown("#### **Bloody Spy Blast Team**")

url1 = "https://linkedin.com/in/said-aliouane-28318012a"
st.sidebar.markdown("**Said Aliouane**   [Lindkedin](%s)" % url1)

url2 = "https://linkedin.com/in/emilienbonhomme"
st.sidebar.markdown("**Emilien Bonhomme**   [Lindkedin](%s)" % url2)

url3 = "https://linkedin.com/in/caroline-le-duigou"
st.sidebar.markdown("**Caroline Le Duigou**   [Lindkedin](%s)" % url3)

url4 = "https://linkedin.com/in/paul-tessier-57b97811a"
st.sidebar.markdown("**Paul Tessier**   [Lindkedin](%s)" % url4)


st.sidebar.markdown("""  """)
st.sidebar.markdown("""  """)
st.sidebar.markdown("""  """)

st.sidebar.markdown("Project Supervisor : **LOUIS** from Datascientest")


#####--------------------------PAGE INTRODUCTION--------------------------#####


if sidebar == "Introduction":
    intro()

#####-----------------------------PAGE DATAVIZ----------------------------#####

elif sidebar == "DataViz":
    dataviz()
    
    
#####------------------------PAGE MACHINE LEARNING------------------------#####
elif sidebar == "Machine Learning":
    Fonction_ML()


#####--------------------------PAGE DEEP LEARNING-------------------------#####

elif sidebar == "Deep Learning":
    ma_fonction_dl()

#####--------------------------PAGE CONCLUSION -------------------------#####

elif sidebar == "Conclusion":
    conclusion()
