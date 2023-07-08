# AI-Smart-Attendance-System

```
This is an advanced Face Recognition Web App that gives you a real times demo of AI Based Smart Attendance System using Streamlit.
```

## Overview of the App

<img src="langchain-text-summarization.jpg" width="75%">

- Accepts a paragraph of text as the input text (to be summarized) using Streamlit's `st.text_input()`
- Text is split into chunks via `CharacterTextSplitter()` along with its `split_text()` method
- Document is generated via `Document()
- Text summarization is achieved using `load_summarize_chain()` by applying the `run()` method on the input `docs`.

## Demo App

![](https://github.com/Anas436/Text-Summarization-App/blob/main/text.gif)

## Installation

To install the repository, please clone this repository and install the requirements:

```
pip install -r requirements.txt
```

## Usage

To use the application, run the `main.py` file with the streamlit CLI (after having installed streamlit): 

```
streamlit run app.py
```
