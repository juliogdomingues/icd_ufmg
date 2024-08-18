import streamlit as st
import pandas as pd
import plotly.express as px

# Cache para otimizar o carregamento de dados
@st.cache_data
def load_data():
    global_data = pd.read_csv("https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/global-data-on-sustainable-energy.csv")
    share_electricity_renewables = pd.read_csv("https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/renewable_energy/04%20share-electricity-renewables.csv")
    countries_by_continents = pd.read_csv("https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/Countries%20by%20continents.csv")
    return global_data, share_electricity_renewables, countries_by_continents

# Carregar dados
global_data, share_electricity_renewables, countries_by_continents = load_data()

# Preparar os dados
def prepare_data(df_renewables, df_continents):
    df_renewables = df_renewables.drop(columns=['Code'])
    df_renewables = df_renewables.rename(columns={'Entity': 'Country'})
    df_renewables = df_renewables.merge(df_continents)
    df_renewables = df_renewables[df_renewables['Renewables (% electricity)'] != 0]
    df_renewables = df_renewables.query('Year >= 2000 and Year <= 2021')
    return df_renewables

# Preparar DataFrame
share_electricity_renewables = prepare_data(share_electricity_renewables, countries_by_continents)

# Título do App
st.title("Análise de Fontes de Energia Renovável Global")

# Mostrar os dados
st.subheader("Dados Carregados")
st.dataframe(global_data.head())

# Seção interativa de análise
st.subheader("Gráfico de Fontes Renováveis por País")

# Selecionar país
selected_country = st.selectbox("Escolha um país:", share_electricity_renewables["Country"].unique())

# Filtrar dados por país
filtered_data = share_electricity_renewables[share_electricity_renewables["Country"] == selected_country]

# Criar gráfico interativo
fig = px.line(filtered_data, x="Year", y="Renewables (% electricity)", title=f"Participação de Energias Renováveis no {selected_country}")
st.plotly_chart(fig)

# Seção para dados adicionais
st.subheader("Outros Dados de Energia")
st.write("Selecione outras fontes de energia para explorar mais gráficos.")

# Adicione mais gráficos ou análises conforme necessário
