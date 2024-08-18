import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint
from sklearn.linear_model import RidgeCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def carregar_dados():
    data = pd.read_csv("https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/global-data-on-sustainable-energy.csv")
    share_electricity_renewables = pd.read_csv('https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/renewable_energy/04%20share-electricity-renewables.csv')
    countries_by_continents = pd.read_csv("https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/Countries%20by%20continents.csv")
    hydro_energy_participation_per_country = pd.read_csv('https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/renewable_energy/07%20share-electricity-hydro.csv')
    wind_energy_participation_per_country = pd.read_csv('https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/renewable_energy/11%20share-electricity-wind.csv')
    solar_energy_participation_per_country = pd.read_csv('https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/renewable_energy/15%20share-electricity-solar.csv')
    biofuel_production_per_country = pd.read_csv('https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/renewable_energy/16%20biofuel-production.csv')
    return data, share_electricity_renewables, countries_by_continents, hydro_energy_participation_per_country, wind_energy_participation_per_country, solar_energy_participation_per_country, biofuel_production_per_country

def prepare_data(share_electricity_renewables, countries_by_continents):
    share_electricity_renewables = share_electricity_renewables.drop(columns=['Code'])
    share_electricity_renewables = share_electricity_renewables.rename(columns={'Entity': 'Country'})
    share_electricity_renewables = share_electricity_renewables.merge(countries_by_continents)
    share_electricity_renewables = share_electricity_renewables[share_electricity_renewables['Renewables (% electricity)'] != 0]
    share_electricity_renewables = share_electricity_renewables.query('Year >= 2000 and Year <= 2021')
    average_share_electricity_renewables_per_continent = (share_electricity_renewables.groupby(['Continent', 'Year'])
                                                          ['Renewables (% electricity)'].mean().round(2).reset_index())
    return average_share_electricity_renewables_per_continent, share_electricity_renewables

def plot_global_renewables(average_share_electricity_renewables_per_continent):
    st.header("Evolução da Participação de Fontes Renováveis na Matriz Energética por Continente")
    fig = px.line(average_share_electricity_renewables_per_continent, x='Year', y='Renewables (% electricity)',
                  color='Continent',
                  color_discrete_sequence=px.colors.qualitative.T10,
                  labels={"Continent": "Continente"}, width=700, height=700)
    fig.update_layout(
        title_text='<b>Evolução da participação de fontes renováveis<br>na matriz energética de cada continente (2000 - 2021)<b>',
        title_x=0.5, font_size=12, xaxis_title='Ano', yaxis_title='Porcentagem (%)')
    st.plotly_chart(fig)

def plot_world_renewables(share_electricity_renewables):
    st.header("Evolução da Participação de Fontes Renováveis na Matriz Energética Mundial")
    average_share_electricity_renewables_per_year = (share_electricity_renewables.groupby('Year')
                                                     ['Renewables (% electricity)'].mean().round(2).reset_index())
    fig = px.line(average_share_electricity_renewables_per_year, x='Year', y='Renewables (% electricity)', width=700, height=700)
    fig.update_layout(
        title_text='<b>Evolução da participação de fontes renováveis<br>na matriz energética mundial (2000 - 2021)<b>',
        title_x=0.5, font_size=12, xaxis_title='Ano', yaxis_title='Porcentagem (%)')
    fig.update_traces(line_color='mediumseagreen')
    st.plotly_chart(fig)

def plot_continent_renewables(data, countries_by_continents):
    st.header("Consumo de Energia Provinda de Fontes Renováveis por Continente")
    global_data = data.rename(columns={'Entity': 'Country'})
    global_data = pd.merge(global_data, countries_by_continents)
    global_data['Financial flows to developing countries (US $)'] = global_data['Financial flows to developing countries (US $)'].div(1e9)
    renewable_energy_in_final_consumption = global_data[['Country', 'Year', 'Renewable energy share in the total final energy consumption (%)', 'Continent']]
    renewable_energy_in_final_consumption = renewable_energy_in_final_consumption.dropna()
    renewable_energy_in_final_consumption_per_continent = (renewable_energy_in_final_consumption.groupby('Continent')
                                                           ['Renewable energy share in the total final energy consumption (%)']
                                                           .agg(['mean', 'median']).round(2).sort_values(by='mean', ascending=True).reset_index())
    fig = px.bar(renewable_energy_in_final_consumption_per_continent, orientation='h', x='mean', y='Continent', color='Continent', color_discrete_sequence=px.colors.qualitative.Prism, width=1500, height=700)
    fig.update_layout(title_text='<b>Consumo de energia provindo de fontes renováveis<b>', title_x=0.5, font_size=15, showlegend=False)
    fig.update_layout(xaxis_title='Percentual do total consumido (%)', yaxis_title='Continente')
    st.plotly_chart(fig)

def plot_specific_choropleth_map(df, countries_names_column, color_column, user_title, user_subtitle):
    min_value_color_column = df[color_column].min()
    max_value_color_column = df[color_column].max()
    fig = px.choropleth(df, locations=countries_names_column, locationmode="country names", color=color_column, animation_frame="Year",
                        color_continuous_scale="YlGnBu", range_color=(min_value_color_column, max_value_color_column))
    fig.update_geos(projection_type="natural earth")
    fig.update_layout(title=user_title, title_x=0.5, font_size=15, coloraxis_colorbar={"title": f'{user_subtitle}'}, height=700)
    return fig

def plot_specific_line_chart(df, x_axis, y_axis, user_title, x_axis_user_title, y_axis_user_title):
    fig = px.line(df, x=x_axis, y=y_axis, color_discrete_sequence=['mediumseagreen'])
    fig.update_layout(title=user_title, title_x=0.5, font_size=15, xaxis_title=x_axis_user_title, yaxis_title=y_axis_user_title)
    return fig

def plot_map_hydro_energy(hydro_energy_participation_per_country):
    st.header("Mapa Global da Produção de Energia Hidrelétrica")
    global_hydro_energy_participation = hydro_energy_participation_per_country.query('Entity == "World" and Year >= 2000 and Year <= 2021')[['Year', 'Hydro (% electricity)']]
    hydro_energy_participation_per_country = hydro_energy_participation_per_country.dropna().round(2)
    hydro_energy_participation_per_country.sort_values(by='Year', ascending=True, inplace=True)
    hydro_energy_participation_per_country = hydro_energy_participation_per_country.query('Year >= 2000 and Year <= 2021')
    fig = plot_specific_choropleth_map(hydro_energy_participation_per_country, 'Entity', 'Hydro (% electricity)', '<b>Taxa de participação de fontes de energia hidrelétricas na matriz de cada país (2000-2021)<b>', 'Porcentagem (%)')
    st.plotly_chart(fig)

def plot_line_hydro_energy(global_hydro_energy_participation):
    st.header("Taxa Média Global de Participação de Fontes de Energia Hidrelétricas nas Matrizes dos Países (2000-2021)")
    fig = plot_specific_line_chart(global_hydro_energy_participation, 'Year', 'Hydro (% electricity)', '<b>Taxa média global de participação de fontes de energia hidrelétricas nas matrizes dos países (2000-2021)<b>', 'Ano', 'Porcentagem (%)')
    st.plotly_chart(fig)

def plot_map_solar_energy(solar_energy_participation_per_country):
    st.header("Mapa Global da Produção de Energia Solar")
    global_solar_energy_participation = solar_energy_participation_per_country.query('Entity == "World" and Year >= 2000 and Year <= 2021')[['Year', 'Solar (% electricity)']]
    solar_energy_participation_per_country = solar_energy_participation_per_country.dropna().round(2)
    solar_energy_participation_per_country.sort_values(by='Year', ascending=True, inplace=True)
    solar_energy_participation_per_country = solar_energy_participation_per_country.query('Year >= 2000 and Year <= 2021')
    fig = plot_specific_choropleth_map(solar_energy_participation_per_country, 'Entity', 'Solar (% electricity)', '<b>Taxa de participação de energia solar na matriz de cada país (2000-2021)<b>', 'Porcentagem (%)')
    st.plotly_chart(fig)

def plot_line_solar_energy(global_solar_energy_participation):
    st.header("Taxa Média Global de Participação de Fontes de Energia Solar nas Matrizes dos Países (2000-2021)")
    fig = plot_specific_line_chart(global_solar_energy_participation, 'Year', 'Solar (% electricity)', '<b>Taxa média global de participação de fontes de energia solares nas matrizes dos países (2000-2021)<b>', 'Ano', 'Porcentagem (%)')
    st.plotly_chart(fig)

def plot_map_wind_energy(wind_energy_participation_per_country):
    st.header("Mapa Global da Produção de Energia Eólica")
    global_wind_energy_participation = wind_energy_participation_per_country.query('Entity == "World" and Year >= 2000 and Year <= 2021')[['Year', 'Wind (% electricity)']]
    wind_energy_participation_per_country = wind_energy_participation_per_country.dropna().round(2)
    wind_energy_participation_per_country.sort_values(by='Year', ascending=True, inplace=True)
    wind_energy_participation_per_country = wind_energy_participation_per_country.query('Year >= 2000 and Year <= 2021')
    fig = plot_specific_choropleth_map(wind_energy_participation_per_country, 'Entity', 'Wind (% electricity)', '<b>Taxa de participação de energia eólica na matriz energética de cada país (2000-2021)<b>', 'Porcentagem (%)')
    st.plotly_chart(fig)

def plot_line_wind_energy(global_wind_energy_participation):
    st.header("Taxa Média Global de Participação de Fontes de Energia Eólica nas Matrizes dos Países (2000-2021)")
    fig = plot_specific_line_chart(global_wind_energy_participation, 'Year', 'Wind (% electricity)', '<b>Taxa média global de participação de fontes de energia eólicas nas matrizes dos países (2000-2021)<b>', 'Ano', 'Porcentagem (%)')
    st.plotly_chart(fig)

def plot_energy_consumption(df, countries_by_continents):
    st.header("Uso de Energias Renováveis e Fósseis")
    df1 = df[['Entity', 'Year', 'Electricity from fossil fuels (TWh)', 'Electricity from renewables (TWh)']].groupby('Entity').sum()
    top_renew = df1.nlargest(10, 'Electricity from renewables (TWh)').reset_index()['Entity']
    df_comp = (df[df['Entity'].isin(top_renew)].rename(columns={'Electricity from renewables (TWh)': 'Renovável', 'Electricity from fossil fuels (TWh)': 'Fóssil'})[["Entity", "Year", "Renovável", "Fóssil"]])
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    sns.lineplot(x='Year', y='Renovável', data=df_comp, hue='Entity', ax=ax[0])
    ax[0].set_title('Renováveis')
    
    sns.lineplot(x='Year', y='Fóssil', data=df_comp, hue='Entity', ax=ax[1])
    ax[1].set_title('Fósseis')
  
    plt.tight_layout()
    st.pyplot(plt)

def plot_scatter_fossil_renewable(df):
    st.header("Energia Fóssil x Renovável")
    df_comp = df.rename(columns={'Electricity from renewables (TWh)': 'Renovável', 'Electricity from fossil fuels (TWh)': 'Fóssil'}) [["Entity", "Year", "Renovável", "Fóssil"]]
    
    # Define df1 dentro da função
    df1 = df[['Entity', 'Year', 'Electricity from fossil fuels (TWh)', 'Electricity from renewables (TWh)']].groupby('Entity').sum()
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.scatterplot(data=df_comp, x='Fóssil', y='Renovável', hue='Year', ax=ax[0])
    ax[0].set_title('Fóssil x Renováveis')
    ax[0].set_xlabel('Energia Fóssil (TWh)')
    ax[0].set_ylabel('Energia Renovável (TWh)')

    top_renew = df1.nlargest(10, 'Electricity from renewables (TWh)').reset_index()['Entity']
    sns.scatterplot(data=df_comp[df_comp['Entity'].isin(top_renew)], x='Fóssil', y='Renovável', hue='Entity', ax=ax[1])

    st.pyplot(fig)

def plot_gdp_vs_renewable(df):
    st.header("PIB per capita x Uso de Energia Renovável")
    media_paises = (df[['Entity','Year','Renewable energy share in the total final energy consumption (%)','gdp_per_capita']]
                   .dropna().groupby('Entity').mean().reset_index())
    fig = px.scatter(data_frame=media_paises, x='gdp_per_capita', y='Renewable energy share in the total final energy consumption (%)',
                     title='PIB per capita X Uso de energia renovável',
                     labels={'gdp_per_capita': 'PIB Per Capita (U$)','Renewable energy share in the total final energy consumption (%)':'Uso de energia renovável (%)'},
                     width=1080, hover_name='Entity')
    st.plotly_chart(fig)

def plot_gdp_vs_renewable_year(df, year):
    st.header(f"PIB per capita x Uso de Energia Renovável em {year}")
    paises_ano = df[['Entity','Year','Renewable energy share in the total final energy consumption (%)','gdp_per_capita']][df['Year']==year]
    fig = px.scatter(data_frame=paises_ano, x='gdp_per_capita', y='Renewable energy share in the total final energy consumption (%)',
                     title=f'PIB per capita X Uso de energia renovável ({year})',
                     labels={'gdp_per_capita': 'PIB Per Capita (U$)','Renewable energy share in the total final energy consumption (%)':'Uso de energia renovável (%)'},
                     width=1080, hover_name='Entity', color_discrete_sequence=['red'])
    st.plotly_chart(fig)

def plot_regression_consumo_gdp(dfBR):
    st.header("Regressão: PIB per Capita vs Consumo de Energia Total per Capita")
    X_total = dfBR['gdp_per_capita']
    y_total = dfBR['Primary energy consumption per capita (kWh/person)']
    X_total = sm.add_constant(X_total)
    model_total = sm.OLS(y_total, X_total).fit()

    plt.figure(figsize=(10, 6))
    plt.plot(dfBR['gdp_per_capita'], y_total, 'o', label="Dados Observados")
    plt.plot(dfBR['gdp_per_capita'], model_total.fittedvalues, 'r--', label="Ajuste do Modelo")
    plt.xlabel('PIB per Capita')
    plt.ylabel('Consumo de Energia Total per Capita (kWh/person)')
    plt.legend(loc='best')
    plt.title('Regressão Linear: PIB per Capita vs Consumo de Energia Total per Capita')
    st.pyplot(plt)

def main():
    st.title("Análise de Fontes de Energia Renováveis")
    
    data, ser, cc, hep, wep, sep, bpc = carregar_dados()
    
    average_share_electricity_renewables_per_continent, share_electricity_renewables = prepare_data(ser, cc)
    
    plot_global_renewables(average_share_electricity_renewables_per_continent)
    
    plot_world_renewables(share_electricity_renewables)
    
    plot_continent_renewables(data, cc)
    
    plot_map_hydro_energy(hep)
    
    plot_line_hydro_energy(hep.query('Entity == "World" and Year >= 2000 and Year <= 2021')[['Year', 'Hydro (% electricity)']])
    
    plot_map_solar_energy(sep)
    
    plot_line_solar_energy(sep.query('Entity == "World" and Year >= 2000 and Year <= 2021')[['Year', 'Solar (% electricity)']])
    
    plot_map_wind_energy(wep)
    
    plot_line_wind_energy(wep.query('Entity == "World" and Year >= 2000 and Year <= 2021')[['Year', 'Wind (% electricity)']])
    
    plot_energy_consumption(data, cc)
    
    plot_scatter_fossil_renewable(data)
    
    plot_gdp_vs_renewable(data)

    selected_year = st.slider("Selecione um ano para o gráfico de PIB vs. Renováveis:", min_value=2000, max_value=2021, value=2020)
    plot_gdp_vs_renewable_year(data, selected_year)
    
    plot_regression_consumo_gdp(data[data['Entity'] == 'Brazil'])

if __name__ == "__main__":
    main()