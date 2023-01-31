import streamlit as st
from PIL import Image

#####--------------------------PAGE INTRODUCTION--------------------------#####

def conclusion():
    # st.markdown('<style>h1{color: #A52A2A;font-size: 70px}</style> ', unsafe_allow_html=True) # définit le style de tous les titres de la page
    # st.markdown('<style>h2{color: #00008B;font-size: 60px}</style> ', unsafe_allow_html=True) # définit le style de tous les sous-titres de la page

    st.markdown("<h1 style='text-align: left; color: #A52A2A;'>Conclusion</h1>", unsafe_allow_html=True)

    st.markdown("""  """)
    st.markdown("""  """)
    st.markdown("""  """)

    # st.markdown("""**Performances**""")
    st.markdown("<h2 style='text-align: left; color: #00008B;'>Performances</h2>", unsafe_allow_html=True)

    results = 'resources/results_tab_60x60_selected_filters.png'
    # results = 'resources/results_tab_60x60.png'
    # results = 'resources/results_tab_all.png'
    img = Image.open(results)
    st.image(img, caption='')

    st.markdown("""The support Vector Machine (SVM) and Random Forest (RF) machine learning models present scores peaking
    at 87% classification score (accuracy). RFs have a greater tendency to over-train and hyperparameter optimization research 
    is needed. Despite the use of several image segmentation methods, raw flatten RGB images seem to give the best results, allowing
    the best computation time/scoring trade off. The best classification was obtained for 75% of blood cells tested, leaving 
    poor scores for monocytes and immature granulocytes.""")

    st.markdown("""The use of convolutional neural networks strongly improved the class prediction scores. Indeed  the best 
    score reached 95% accuracy by using the pre-trained VGG16 model on the imagenet dataset and retrained on the flattened RGB 
    dataset. The use of deeplearning improved the classification of  monocytes. The only class that still presents concerns is
    the immature granulocytes Ig, which seems logical since it is an heterogeneous class.""")

    st.markdown("""Augmenting the dataset with a data generator did not improve performance for all models.""")

    st.markdown("""The scores we obtained with deep learning (95%) are honorable compared to the  state of the art of blood cell
    classification. Indeed Acevedo and colleagues (2019) achieved similar scores of 96% using the pre-trained VGG16 and InceptionV3 models..""")

    st.markdown("""  """)
    st.markdown("""  """)
    st.markdown("""  """)


    # st.markdown("""**Next**""")
    st.markdown("<h2 style='text-align: left; color: #00008B;'>What's next ?</h2>", unsafe_allow_html=True)

    st.markdown("""1. Necessity of fine tuning on the parameters of our pre-trained models, the VGG16 and VGG19 models,
    to adapt their layers to our data and improve their learning. """)

    st.markdown(" ")
    st.markdown("""2. In this first part of the project, models were trained to recognize blood cell types. Knowing that the density 
    and relative abundance of blood cells in the smear is crucial for the diagnosis of many pathologies, it would be necessary to 
    identify and count the cells. The method would first consist of identifying the bounding boxes of the different cells in the image
     using Faster R-CNN or YOLO V4 models and then classifying the cells..""")

    st.markdown(" ")

    c1,c2 = st.columns(2)

    # Affichage Photo
    image1 = "resources/Faster_RCNN1.webp"
    c1.image(image1, caption=None, width=400, use_column_width=300, clamp=False, channels="RGB", output_format="auto")
    image2 = "resources/YOLO4_cell_detection.png"
    c2.image(image2, caption=None, width=400, use_column_width=300, clamp=False, channels="RGB", output_format="auto")
    c1.markdown("""**Faster_R_CNN**""")
    c2.markdown("""**YOLO4**""")

    st.markdown("""3. It would be interesting to study the singularities of cell type that the layers of our neural networks detect,
     thanks to the Grad-CAM library of PyTorch which proposes to visualize the parts of images which particularly activate our network.
     The comparison with the analysis of misclassified images would allow us to identify insignificant patterns that activate our 
     neurons, and to correct upstream by a segmentation filter.""")

    st.markdown(" ")
    image3 = "resources/grad_cam_lymphocytes.png"
    st.image(image3, caption=None, width=600, use_column_width=300, clamp=False, channels="RGB", output_format="auto")
    st.markdown("""**Visualization of gradient class activation maps (Grad-CAM) from different CNN architecture**""")


    # url1 = "https://data.mendeley.com/datasets/snkd93bnjr/1"
    # st.markdown('**Access to leukemia dataset** [link](%s)' % url1, unsafe_allow_html=True)
    # url2 = "https://www.kaggle.com/eugeneshenderov/acute-promyelocytic-leukemia-apl"
    # st.markdown('**Access to leukemia dataset 2** [link](%s)' % url2, unsafe_allow_html=True)


    st.markdown("""  """)
    st.markdown("""  """)
    st.markdown("""  """)
    st.markdown("""  """)
    st.markdown("""  """)
    st.markdown("""  """)
    st.markdown("""  """)
    st.markdown("""  """)
    st.markdown("""  """)

    col1,col2, col3 = st.columns([1, 1, 1])
    img = Image.open('resources/thank_you.png')
    col2.image(img, caption='')

