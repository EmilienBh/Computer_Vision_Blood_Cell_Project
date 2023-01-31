import streamlit as st

#####--------------------------PAGE INTRODUCTION--------------------------#####

def intro():
    st.markdown('<style>h1{color: #A52A2A;font-size: 70px}</style> ', unsafe_allow_html=True) # définit le style de tous les titres de la page
    st.markdown('<style>h2{color: #00008B;font-size: 60px}</style> ', unsafe_allow_html=True) # définit le style de tous les sous-titres de la page

    st.title("Bloody Spy Blast")

    #st.subheader("Introduction")
    #st.markdown('<style>h1{color: blue;font-size: 60px}</style> ', unsafe_allow_html=True)

    st.subheader("Automatic recognition of human blood cell types")
    # st.markdown('<style>h1{color: red;font-size: 70px}</style>, ',  unsafe_allow_html=True)

    # Affichage Photo
    image = "resources/blood_cells_in_vein.jpg"
    st.image(image, caption=None, width=700, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

    st.markdown(""" <style> .font {
      font-size:100px;} 
      </style> """, unsafe_allow_html=True)

    st.markdown("""**Introduction**""")


    st.markdown("""Blood is an essential element for human survival. It allows the supply of oxygen and nutrients to the
    organs, blood coagulation but also the immune defenses ‘ transport against bacterial and viral attacks. Blood contains
    several cell types that perform these functions. Thus the appareance of disturbances in blood composition is a good
    marker of pathologies presence. <style> .font {font-size:100px;}</style> """, unsafe_allow_html=True)

    st.markdown("""Today hematological diseases are diagnosed in more than 80% of cases thanks to 
    quantitative and qualitative analyzes of the different cell types. However, the morphological differentiation of normal
    and abnormal blood cell types is a difficult task requiring significant expertise.  
    In order to overcome the lack of expertise in certain medical circles, the creation of auto diagnostic models 
    of human blood pathologies would therefore be an interesting tool to explore.""")
    st.markdown("""Therefore this project consists of developing computer vision models capable of identifying the 
    different types of blood cells through the analysis of human blood cells collected by blood smears. The training of 
    these models will be carried out on a database of blood cells from healthy subjects.""")

    st.markdown("""  """)
    st.markdown("""  """)
    st.markdown("""  """)
    st.markdown("""**Blood cell types description**""")

    st.markdown("""Peripheral blood contains three main cell types : the **erythrocytes (red blood cells)** which are responsible
       of the transport of oxygen to the organs, the **thrombocytes (platelets)** allowing blood coagulation and the **leukocytes
        (white blood cells)** protecting the body against viral and bacterial invasions. The differentiation of those cell types
         is possible by morphological analyses thanks to specific coloration methods like the  May-Grün coloration. """)

    st.markdown("""From a structural point of view, the leucocytes can be subdivided into three major classes : the **granulocytes**,
     the **lymphocytes** and the **monocytes**. The class of granulocytes itself includes several subtypes according to their coloration:
      the **neutrophil** granulocytes, the **eosinophilic** granulocytes and the **basophilic** granulocytes.""")

    st.markdown("""During disease onset, it is also possible to find **immature granulocytes** that contain the classes of myeloblast, 
    promyelocyte, myelocyte and metamyelocyte. Usually they are in the spinal bone but a presence of a high level of those cells 
    in the blood could be the sign of a cancer. That's why their detection and their quantification in the blood are important 
    for disease diagnosis. """)


    # Affichage Photo description cellules
    image1 = "resources/blood_cell_types.jpg"
    image2 = "resources/immature_granulocytes.gif"
    c1, c2 = st.columns([1,1])
    c1.image(image1, caption=None, width=700, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    c2.image(image2, caption=None, width=700, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

