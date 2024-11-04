# No arquivo `streamlit_app.py`
import streamlit as st

# Executa o código do arquivo `graphic.py`
with open("graphic.py") as f:
    exec(f.read())
