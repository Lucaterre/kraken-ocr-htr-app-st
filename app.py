#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streamlit interface for OCR/HTR with Kraken"""
import os
import datetime
import random

import streamlit as st

from lib.constants import CONFIG_METADATA
from lib.display_utils import (display_baselines,
                               display_baselines_with_text,
                               prepare_segments,
                               open_image)
from lib.kraken_utils import (load_model_seg,
                              load_model_rec,
                              segment_image,
                              recognize_text)

# === PAGE CONFIGURATION ===
st.set_page_config(layout="wide")

# === I/O UTILS ===
def get_real_path(path: str) -> str:
    """Get absolute path of a file."""
    return os.path.join(os.path.dirname(__file__), path)


def load_random_example_image(folder_path: str):
    """Load a random image from a folder."""
    images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('jpg', 'jpeg'))]
    return random.choice(images) if images else None


def write_temporary_model(file_path, custom_model_loaded):
    """Write a temporary model to disk."""
    with open(get_real_path(file_path), "wb") as file:
        file.write(custom_model_loaded.getbuffer())


def load_model_seg_cache(model_path):
    return load_model_seg(model_path)


def load_model_rec_cache(model_path):
    return load_model_rec(model_path)


MODEL_SEG_BLLA = load_model_seg_cache(get_real_path("data/default/blla.mlmodel"))


def load_models(model_rec_in, model_seg_in=None):
    """Generic bridge to load models.
    """
    if model_rec_in is not None:
        try:
            model_rec_out = load_model_rec_cache(model_rec_in)
        except Exception as e:
            st.error(f" ❌ Modèle de reconnaissance non chargé. Erreur : {e}")
            return None, None
    else:
        st.error(" ❌ Modèle de reconnaissance non trouvé.")
        return None, None
    if model_seg_in is not None:
        try:
            model_seg_out = load_model_seg_cache(model_seg_in)
        except Exception as e:
            st.error(f" ❌ Modèle de segmentation non chargé. Erreur : {e}")
            return None, None
    else:
        model_seg_out = MODEL_SEG_BLLA
    return model_rec_out, model_seg_out

# === MODELS EXAMPLES ===


endp_model_rec, endp_model_seg = load_models(model_rec_in=get_real_path("data/endp/models/e-NDP_V7.mlmodel"),
                                             model_seg_in=get_real_path("data/endp/models/e-NDP-seg_V3.mlmodel"))
lectaurep_model_rec = load_model_rec(get_real_path("data/lectaurep/models/lectaurep_base.mlmodel"))
catmus_model_rec = load_model_rec(get_real_path("data/catmus/models/catmus-tiny.mlmodel"))

# === MODELS EXAMPLES CONFIGURATION ===
DEFAULT_CONFIG = {
    'endp': {
        'model_rec': endp_model_rec,
        'model_seg': endp_model_seg,
        'example_images': get_real_path("data/endp/images")
    },
    'lectaurep': {
        'model_rec': lectaurep_model_rec,
        'model_seg': None,
        'example_images': get_real_path("data/lectaurep/images")
    },
    'catmus':{
        'model_rec': catmus_model_rec,
        'model_seg': None,
        'example_images': get_real_path("data/catmus/images")
    }
}


# === USER INTERFACE ===
st.title("📜🦑 Reconnaissance de Texte (OCR/HTR) avec Kraken")
st.markdown("[![https://img.shields.io/badge/Kraken_version-5.2.9-orange](https://img.shields.io/badge/Kraken_version-5.2.9-orange)](https://github.com/mittagessen/kraken)")
st.markdown(
    """
    *⚠️ Cette application est à visée pédagogique ou à des fins de tests uniquement. 
    L'auteur se dégage de toutes responsabilités quant à son usage pour la production.*
    """
)
st.markdown(
    """
    ##### 🔗 Ressources :
    - 📂 Données de tests ou d'entraînement dans l'organisation [HTR United](https://htr-united.github.io/index.html)
    - 📦 Modèles (mlmodel) à tester sur le groupe [OCR/HTR Zenodo](https://zenodo.org/communities/ocr_models/records?q=&l=list&p=1&s=10&sort=newest)
    - 🛠 Évaluer vos prédictions avec l'application [KaMI (Kraken as Model Inspector)](https://huggingface.co/spaces/lterriel/kami-app)
    """,
    unsafe_allow_html=True
)

# Configuration choices
st.sidebar.header("📁 Configuration HTR")

st.sidebar.markdown('---')
button_placeholder = st.sidebar.empty()
success_loaded_models_msg_container = st.sidebar.empty()
download_predictions_placeholder = st.sidebar.empty()
st.sidebar.markdown('---')

config_choice = st.sidebar.radio(
    "Choisissez une configuration :", options=["Custom", "endp (exemple)", "lectaurep (exemple)", "catmus (exemple)"]
)

config_choice_placeholder = st.sidebar.empty()
info_title_desc = st.sidebar.empty()
place_metadata = st.sidebar.empty()
map_config_choice = {
    "Custom": "Custom",
    "endp (exemple)": "endp",
    "lectaurep (exemple)": "lectaurep",
    "catmus (exemple)": "catmus"
}
config_choice = map_config_choice[config_choice]
flag_rec_model = False
flag_seg_model = False
if config_choice != "Custom":
    config = DEFAULT_CONFIG[config_choice]
    config_choice_placeholder.success(f"Configuration sélectionnée : {CONFIG_METADATA[config_choice]['title']}")
    place_metadata.markdown(CONFIG_METADATA[config_choice]['description'], unsafe_allow_html=True)
    flag_rec_model = True
else:
    st.sidebar.warning("Configuration personnalisée")
    custom_model_seg = st.sidebar.file_uploader(
        "Modèle de segmentation (optionnel)", type=["mlmodel"]
    )
    custom_model_rec = st.sidebar.file_uploader(
        "Modèle de reconnaissance", type=["mlmodel"]
    )
    if custom_model_rec:
        write_temporary_model('tmp/model_rec_temp.mlmodel', custom_model_rec)
        flag_rec_model = True
    if custom_model_seg:
        write_temporary_model('tmp/model_seg_temp.mlmodel', custom_model_seg)
        flag_seg_model = True


# Image choice
flag_image = False
image_source = st.radio("Source de l'image :", options=["Exemple", "Personnalisée"])
info_example_image = st.empty()
info_example_image_description = st.empty()
upload_image_placeholder = st.empty()
col1, col2, col3 = st.columns([1, 1, 1])
image = None
with col1:
    st.markdown("## 🖼 Image Originale")
    st.markdown("---")
    if image_source == "Exemple":
        if config_choice != "Custom":
            example_image_path = load_random_example_image(config["example_images"])
            if example_image_path:
                image = open_image(example_image_path)
                flag_image = True
                info_example_image.info(f"Image d'exemple chargée : {os.path.basename(example_image_path)}")
                info_title_desc.markdown(
                    "<h4>Métadonnées de la configuration</h3>", unsafe_allow_html=True)
                info_example_image_description.markdown(
                    f"Source : {CONFIG_METADATA[config_choice]['examples_info'][os.path.basename(example_image_path)]}",
                    unsafe_allow_html=True)
            else:
                info_example_image.error("Aucune image d'exemple trouvée.")
        else:
            info_example_image.error("Les images d'exemple ne sont pas disponibles pour la configuration personnalisée.")
    else:
        image_file = upload_image_placeholder.file_uploader("Téléchargez votre image :", type=["jpg", "jpeg"])
        if image_file:
            image = open_image(image_file)
            flag_image = True
        else:
            info_example_image.warning("Veuillez télécharger une image.")
    if flag_image:
        st.image(image, use_container_width=True)

# Display the results
col4, col5, col6 = st.columns([1, 1, 1])
if "image" in locals() and flag_rec_model and flag_image:
    button_pred = button_placeholder.button('🚀Lancer la prédiction', key='but_pred')
    if button_pred:
        with st.spinner("⚙️ Chargement des nouveaux modèles..."):
            if config_choice != "Custom":
                model_rec, model_seg = DEFAULT_CONFIG[config_choice]['model_rec'], DEFAULT_CONFIG[config_choice]['model_seg']
            else:
                model_rec = load_model_rec_cache(get_real_path('tmp/model_rec_temp.mlmodel')) if flag_rec_model else None
                model_seg = load_model_seg_cache(get_real_path('tmp/model_seg_temp.mlmodel')) if flag_seg_model else None
            success_loaded_models_msg_container.success("✅️ Configuration OK!")

        with col2:
            st.markdown("## ✂️Segmentation")
            st.markdown("---")
            with st.spinner("⚙️ Segmentation en cours..."):
                baseline_seg = segment_image(image, model_seg)
                baselines, boundaries = prepare_segments(baseline_seg)
            fig1, fig2 = display_baselines(image, baselines, boundaries)
            st.pyplot(fig1)

        with col3:
            st.markdown("## ✍️ Texte")
            st.markdown("---")
            with st.spinner("⚙️ Reconnaissance en cours..."):
                pred = recognize_text(model_rec, image, baseline_seg)
                lines = [record.prediction.strip() for record in pred]
                lines_with_idx = [f"{idx}: {line}" for idx, line in enumerate(lines)]
            st.text_area(label='', value="\n".join(lines), height=570, label_visibility="collapsed")
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        with col4:
            st.markdown("## ✂ Segmentation (Index)")
            st.markdown("---")
            st.pyplot(fig2)

        with col5:
            st.markdown("## ✏ Texte (Index)")
            st.markdown("---")
            st.text_area(label='', value="\n".join(lines_with_idx), height=570, label_visibility="collapsed")

        with col6:
            st.markdown("## 🔎 Texte (Image)")
            st.markdown("---")
            st.pyplot(display_baselines_with_text(image, baselines, lines))

        download_predictions_placeholder.download_button(
            "💾 Télécharger votre prédiction (txt)",
            "\n".join(lines),
            file_name=f"prediction_{date}.txt",
        )
