# Document Segmentation & Recognition with Kraken | Streamlit application

![https://img.shields.io/badge/Python-3.9-blue](https://img.shields.io/badge/Python-3.9-blue)
[![https://img.shields.io/badge/Kraken_version-5.2.9-orange](https://img.shields.io/badge/Kraken_version-5.2.9-orange)](https://github.com/mittagessen/kraken)
![streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

➡️ Try on HuggingFace Spaces 🤗 : https://huggingface.co/spaces/lterriel/kraken-htr-ocr-app

*This is a mini Streamlit application to try, experiment and/or test new segmentation and recognition models trained with Kraken on your own images.*

*This app build with pedagogical purpose to show how to use Kraken models and what are the steps in generic OCR/HTR pipeline before start a real or much bigger project.*

📊 To evaluate the recognition model performance on your data, you can use [KaMI-App](https://github.com/KaMI-tools-project/KaMI-app), a generic metric dashboard app for OCR/HTR models.


### Install & run the app locally

create a virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```