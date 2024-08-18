import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


# Carregando as bases de dados
data = pd.read_csv("https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/global-data-on-sustainable-energy.csv")
share_electricity_renewables = pd.read_csv('https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/renewable_energy/04%20share-electricity-renewables.csv')
countries_by_continents = pd.read_csv("https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/Countries%20by%20continents.csv")
hydro_energy_participation_per_country = pd.read_csv('https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/renewable_energy/07%20share-electricity-hydro.csv')
wind_energy_participation_per_country = pd.read_csv('https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/renewable_energy/11%20share-electricity-wind.csv')
solar_energy_participation_per_country = pd.read_csv('https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/renewable_energy/15%20share-electricity-solar.csv')
biofuel_production_per_country = pd.read_csv('https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/renewable_energy/16%20biofuel-production.csv')

# Função para carregar e transformar dados
def prepare_data():
    global share_electricity_renewables, countries_by_continents

    share_electricity_renewables = share_electricity_renewables.drop(columns=['Code'])
    share_electricity_renewables = share_electricity_renewables.rename(columns={'Entity': 'Country'})
    share_electricity_renewables = share_electricity_renewables.merge(countries_by_continents)

    share_electricity_renewables = share_electricity_renewables[share_electricity_renewables['Renewables (% electricity)'] != 0]
    share_electricity_renewables = share_electricity_renewables.query('Year >= 2000 and Year <= 2021')

    average_share_electricity_renewables_per_continent = (share_electricity_renewables
                                                          .groupby(['Continent', 'Year'])
                                                          ['Renewables (% electricity)']
                                                          .mean()
                                                          .round(2)
                                                          .reset_index())

    return average_share_electricity_renewables_per_continent

# Função para plotar gráficos
def plot_graphs():
    st.title("Análise de Fontes de Energia Renováveis")

    # Preparando os dados
    average_share_electricity_renewables_per_continent = prepare_data()

    # Gráficos
    average_share_electricity_renewables_per_continent_plot = px.line(
        average_share_electricity_renewables_per_continent,
        x='Year',
        y='Renewables (% electricity)',
        color = 'Continent',
        color_discrete_sequence=px.colors.qualitative.T10,
        labels={"Continent": "Continente"},
        width=700,
        height=700
    )

    average_share_electricity_renewables_per_continent_plot.update_layout(
        title_text='<b>Evolução da participação de fontes renováveis<br>na matriz energética de cada continente (2000 - 2021)<b>',
        title_x=0.5,
        font_size=12,
        xaxis_title='Ano',
        yaxis_title='Porcentagem (%)'
    )

    st.plotly_chart(average_share_electricity_renewables_per_continent_plot)
    
    # Gráfico global
    average_share_electricity_renewables_per_year = (share_electricity_renewables
                                                     .groupby('Year')
                                                     ['Renewables (% electricity)']
                                                     .mean()
                                                     .round(2)
                                                     .reset_index())

    average_share_electricity_renewables_per_year_plot = px.line(
        average_share_electricity_renewables_per_year,
        x='Year',
        y='Renewables (% electricity)',
        width=700,
        height=700
    )

    average_share_electricity_renewables_per_year_plot.update_layout(
        title_text='<b>Evolução da participação de fontes renováveis<br>na matriz energética mundial (2000 - 2021)<b>',
        title_x=0.5,
        font_size=12,
        xaxis_title='Ano',
        yaxis_title='Porcentagem (%)'
    )

    average_share_electricity_renewables_per_year_plot.update_traces(line_color='mediumseagreen')
    st.plotly_chart(average_share_electricity_renewables_per_year_plot)

    # Gráfico de barras
    global_data = data.rename(columns={'Entity': 'Country'})

    global_data = pd.merge(global_data, countries_by_continents)

    global_data['Financial flows to developing countries (US $)'] = global_data['Financial flows to developing countries (US $)'].div(1e9)

    renewable_energy_in_final_consumption = global_data[['Country', 'Year', 'Renewable energy share in the total final energy consumption (%)', 'Continent']]
    renewable_energy_in_final_consumption = renewable_energy_in_final_consumption.dropna()

    renewable_energy_in_final_consumption_per_continent = (renewable_energy_in_final_consumption
                                                           .groupby('Continent')
                                                           ['Renewable energy share in the total final energy consumption (%)']
                                                           .agg(['mean', 'median'])
                                                           .round(2)
                                                           .sort_values(by='mean', ascending=True)
                                                           .reset_index())

    renewable_energy_consumption_per_continent_plot = px.bar(
        renewable_energy_in_final_consumption_per_continent,
        orientation='h',
        x='mean',
        y='Continent',
        color='Continent',
        color_discrete_sequence=px.colors.qualitative.Prism,
        width=1500,
        height=700
    )

    renewable_energy_consumption_per_continent_plot.update_layout(
        title_text='<b>Consumo de energia provindo de fontes renováveis<b>', title_x=0.5, font_size=15, showlegend=False)
    renewable_energy_consumption_per_continent_plot.update_layout(xaxis_title='Percentual do total consumido (%)', yaxis_title='Continente')

    st.plotly_chart(renewable_energy_consumption_per_continent_plot)

    # Outros gráficos podem ser adicionados utilizando a mesma metodologia.
    # Por exemplo, para o gráfico "Ajuda financeira recebida para investimento...":

    # Preencher as funções e chamadas para os outros gráficos aqui...

plot_graphs()