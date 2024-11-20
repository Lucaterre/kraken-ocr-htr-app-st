# Document Segmentation & Recognition with Kraken | Streamlit application

![https://img.shields.io/badge/Python-3.9-blue](https://img.shields.io/badge/Python-3.9-blue)
[![https://img.shields.io/badge/Kraken_version-5.2.9-orange](https://img.shields.io/badge/Kraken_version-5.2.9-orange)](https://github.com/mittagessen/kraken)
![streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

➡️ Try on HuggingFace Spaces 🤗 : https://huggingface.co/spaces/lterriel/kraken-htr-ocr-app 

*This is a mini Streamlit application to try, quick experiment and/or test any segmentation and recognition models trained with Kraken on your own images.*

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

### Usage

This application is for now only in French (WIP: English version coming soon).

**For quick testing purposes, the application provides three example configurations (models) from the following repositories:**

- [e-NDP project](https://zenodo.org/records/7575693)
- [Lectaurep](https://zenodo.org/records/6542744)
- [CATMuS-Print [Tiny]](https://zenodo.org/records/10602357)

**Disclaimer:** 

> This project is completely independent and not affiliated with the Kraken project. 
> The models are provided by the respective authors and are used for educational purposes only.<br>Please, don't forget to cite the authors if you use their models in your research.
> The author of this project is not responsible for the content of the models and the results obtained with them. 
> The author disclaims all responsibility if this software is use in production.

**Issues and contributions:**

Feel free to open an issue or contribute with PR to this project :)