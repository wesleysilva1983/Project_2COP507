import streamlit as st
import joblib
import pandas as pd

# Título do aplicativo
st.title("Aplicativo de Predição com Modelo Random Forest")

# Passo 1: Carregar o modelo salvo
@st.cache_resource  # Cache para evitar recarregar o modelo a cada execução
def load_model():
    return joblib.load('rf_model1.joblib')

rf_classifier_loaded = load_model()

# Passo 2: Carregar o arquivo CSV para predição
uploaded_file = st.file_uploader("Envie um arquivo CSV para fazer predições", type=["csv"])

if uploaded_file is not None:
    # Passo 3: Ler o arquivo e preparar os dados
    data = pd.read_csv(uploaded_file)
    
    # Verificar se o arquivo tem as colunas corretas
    st.write("Visualização dos dados carregados:")
    st.write(data.head())  # Mostra as primeiras linhas do arquivo
    
    # Suponha que `data` contenha as colunas necessárias para a predição
    try:
        # Passo 4: Fazer a predição
        y_pred_new = rf_classifier_loaded.predict(data)
        
        # Exibir as previsões
        st.write("Predições para o novo conjunto de dados:")
        st.write(y_pred_new)
        
        # Opcional: Exibir as previsões junto com os dados de entrada
        result_df = data.copy()
        result_df['Previsão'] = y_pred_new
        st.write("Dados com Previsões:")
        st.write(result_df)

        # Opcional: Permitir download do resultado com previsões
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Baixar resultados com previsões",
            data=csv,
            file_name='predicoes.csv',
            mime='text/csv'
        )
        
    except Exception as e:
        st.write("Erro ao fazer predições. Verifique se o arquivo possui o formato correto.")
        st.write(e)
else:
    st.write("Por favor, envie um arquivo CSV para fazer predições.")
