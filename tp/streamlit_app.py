import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Função para carregar os dados (com cache para melhor performance)


@st.cache_data
def load_data():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/global-data-on-sustainable-energy.csv")
    share_electricity_renewables = pd.read_csv(
        'https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/renewable_energy/04%20share-electricity-renewables.csv')
    countries_by_continents = pd.read_csv(
        "https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/Countries%20by%20continents.csv")
    hydro_energy_participation_per_country = pd.read_csv(
        'https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/renewable_energy/07%20share-electricity-hydro.csv')
    wind_energy_participation_per_country = pd.read_csv(
        'https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/renewable_energy/11%20share-electricity-wind.csv')
    solar_energy_participation_per_country = pd.read_csv(
        'https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/renewable_energy/15%20share-electricity-solar.csv')
    return {
        "data": data,
        "share_electricity_renewables": share_electricity_renewables,
        "countries_by_continents": countries_by_continents,
        "hydro_energy_participation_per_country": hydro_energy_participation_per_country,
        "wind_energy_participation_per_country": wind_energy_participation_per_country,
        "solar_energy_participation_per_country": solar_energy_participation_per_country
    }


data = load_data()

# Função para plotar mapas cloropléticos animados


def plot_animated_choropleth_map(df, countries_names_column, color_column, user_title, user_subtitle):
    min_value_color_column = df[color_column].min()
    max_value_color_column = df[color_column].max()

    df_plot = px.choropleth(df,
                            locations=countries_names_column,
                            locationmode="country names",
                            color=color_column,
                            animation_frame="Year",
                            color_continuous_scale="YlGnBu",
                            range_color=(min_value_color_column, max_value_color_column))

    df_plot.update_geos(projection_type="natural earth")

    df_plot.update_layout(title=user_title, font_size=15,
                          coloraxis_colorbar={"title": f'{user_subtitle}'}, height=700)

    st.plotly_chart(df_plot)

# Função para plotar gráfico de linhas


def plot_line_chart_energy_source_use(df, x_axis, y_axis, user_title, x_axis_user_title, y_axis_user_title):
    df_plot = px.line(df, x=x_axis, y=y_axis,
                      color_discrete_sequence=['mediumseagreen'])

    df_plot.update_layout(
        title=user_title,
        font_size=15,
        xaxis_title=x_axis_user_title,
        yaxis_title=y_axis_user_title,
        height=500,
        width=800
    )

    st.plotly_chart(df_plot)

# Seção 1: Análise Global

def analyze_global_data():
    st.title("Análise mundial da energia sustentável")

    # Subseção 1.1: Evolução das fontes limpas na matriz global
    st.header("Evolução das fontes limpas na matriz global")

    share_electricity_renewables = data['share_electricity_renewables']
    countries_by_continents = data['countries_by_continents']

    share_electricity_renewables = share_electricity_renewables.drop(columns=[
                                                                     'Code'])
    share_electricity_renewables = share_electricity_renewables.rename(columns={
                                                                       'Entity': 'Country'})
    share_electricity_renewables = share_electricity_renewables.merge(
        countries_by_continents)
    share_electricity_renewables = share_electricity_renewables[
        share_electricity_renewables['Renewables (% electricity)'] != 0]
    share_electricity_renewables = share_electricity_renewables.query(
        'Year >= 2000 and Year <= 2021')

    # Gráfico de linhas: Evolução por continente
    average_share_electricity_renewables_per_continent = (share_electricity_renewables
                                                          .groupby(['Continent', 'Year'])
                                                          ['Renewables (% electricity)']
                                                          .mean()
                                                          .round(2)
                                                          .reset_index())
    fig = px.line(average_share_electricity_renewables_per_continent, x='Year', y='Renewables (% electricity)',
                  color='Continent', color_discrete_sequence=px.colors.qualitative.T10,
                  title='Evolução da participação de fontes renováveis na matriz energética de cada continente (2000 - 2021)')
    fig.update_layout(xaxis_title='Ano', yaxis_title='Porcentagem (%)')
    st.plotly_chart(fig)

    # Gráfico de linhas: Média global
    average_share_electricity_renewables_per_year = (share_electricity_renewables
                                                     .groupby('Year')
                                                     ['Renewables (% electricity)']
                                                     .mean()
                                                     .round(2)
                                                     .reset_index())
    fig = px.line(average_share_electricity_renewables_per_year, x='Year', y='Renewables (% electricity)',
                  title='Evolução da participação de fontes renováveis na matriz energética mundial (2000 - 2021)')
    fig.update_layout(xaxis_title='Ano', yaxis_title='Porcentagem (%)')
    fig.update_traces(line_color='mediumseagreen')
    st.plotly_chart(fig)

    # Subseção 1.2: Produção bruta de energia por país
    st.header("Produção bruta de energia por país")

    df_renewable = pd.read_csv(
        "https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/renewable_energy/03%20modern-renewable-prod.csv")
    df_renewable = df_renewable.dropna(subset=['Code'])
    df_renewable = df_renewable[df_renewable['Entity'] != 'World']
    df_renewable = df_renewable.query("Year >= 2000 and Year <= 2020")

    temp = df_renewable.groupby('Entity').sum()

    top_wind = (temp
                .nlargest(10, 'Electricity from wind (TWh)')
                .reset_index()['Entity'])

    top_hydro = (temp
                 .nlargest(10, 'Electricity from hydro (TWh)')
                 .reset_index()['Entity'])

    top_solar = (temp
                 .nlargest(10, 'Electricity from solar (TWh)')
                 .reset_index()['Entity'])

    # Gráfico de linhas: Produção de energia por país(vento, água e solar)

    fig_wind = go.Figure()

    for entity in top_wind:
        df_subset = df_renewable[df_renewable['Entity'] == entity]
        fig_wind.add_trace(go.Scatter(x=df_subset['Year'], y=df_subset['Electricity from wind (TWh)'],
                                    mode='lines', name=entity))

    fig_wind.update_layout(
        title='Produção de energia eólica (Top 10 países)',
        xaxis_title='Ano',
        yaxis_title='Produção de eletricidade (TWh)',
        legend_title='Países',
        height=500,
        width=800
    )

    st.plotly_chart(fig_wind)

    # Hydro energy plot
    fig_hydro = go.Figure()

    for entity in top_hydro:
        df_subset = df_renewable[df_renewable['Entity'] == entity]
        fig_hydro.add_trace(go.Scatter(x=df_subset['Year'], y=df_subset['Electricity from hydro (TWh)'],
                                    mode='lines', name=entity))

    fig_hydro.update_layout(
        title='Produção de energia hidrelétrica (Top 10 países)',
        xaxis_title='Ano',
        yaxis_title='Produção de eletricidade (TWh)',
        legend_title='Países',
        height=500,
        width=800
    )

    st.plotly_chart(fig_hydro)

    # Solar energy plot
    fig_solar = go.Figure()

    for entity in top_solar:
        df_subset = df_renewable[df_renewable['Entity'] == entity]
        fig_solar.add_trace(go.Scatter(x=df_subset['Year'], y=df_subset['Electricity from solar (TWh)'],
                                    mode='lines', name=entity))

    fig_solar.update_layout(
        title='Produção de energia solar (Top 10 países)',
        xaxis_title='Ano',
        yaxis_title='Produção de eletricidade (TWh)',
        legend_title='Países',
        height=500,
        width=800
    )

    st.plotly_chart(fig_solar)

    # Subseção 1.3: Padrões entre fontes fósseis e limpas
    st.header("Comparação entre fontes fósseis e limpas")

    df1 = pd.read_csv(
        "https://raw.githubusercontent.com/souza-marcos/ICD_Projeto/main/data/global-data-on-sustainable-energy.csv")

    temp = df1[['Entity', 'Year', 'Electricity from fossil fuels (TWh)', 'Electricity from renewables (TWh)']].groupby(
        'Entity').sum()
    top_renew = (temp
                 .nlargest(10, 'Electricity from renewables (TWh)')
                 .reset_index()['Entity'])

    df_comp = (df1[df1['Entity'].isin(top_renew)]
               .rename(columns={'Electricity from renewables (TWh)': 'Renovável', 'Electricity from fossil fuels (TWh)': 'Fóssil'})
               [["Entity", "Year", "Renovável", "Fóssil"]])

    fig_renovavel = go.Figure()
    for entity in df_comp['Entity'].unique():
        df_subset = df_comp[df_comp['Entity'] == entity]
        fig_renovavel.add_trace(go.Scatter(x=df_subset['Year'], y=df_subset['Renovável'],
                                        mode='lines', name=entity))

    fig_renovavel.update_layout(
        title='Renováveis (Top 10 países)',
        xaxis_title='Ano',
        yaxis_title='Energia Renovável',
        yaxis=dict(range=[0, 5300]),  # Set the y-axis limit
        height=500,
        width=750  # Adjust size for single plot
    )

    st.plotly_chart(fig_renovavel)

    fig_fossil = go.Figure()

    for entity in df_comp['Entity'].unique():
        df_subset = df_comp[df_comp['Entity'] == entity]
        fig_fossil.add_trace(go.Scatter(x=df_subset['Year'], y=df_subset['Fóssil'],
                                        mode='lines', name=entity))

    # Update layout for Fósseis plot
    fig_fossil.update_layout(
        title='Fósseis (Top 10 países)',
        xaxis_title='Ano',
        yaxis_title='Energia Fóssil',
        yaxis=dict(range=[0, 5300]),  # Set the y-axis limit
        height=500,
        width=750  # Adjust size for single plot
    )

    st.plotly_chart(fig_fossil)



# Seção 2: Consumo por Localidade

def consumption_by_location():
    st.title("Consumo de energia por localização")

    # Subseção 2.1: Consumo de energia renovável por continente
    st.header("Consumo de energia renovável por continente")
    global_data = data['data']
    countries_by_continents = data['countries_by_continents']

    global_data.rename(columns={'Entity': 'Country'}, inplace=True)
    global_data = pd.merge(global_data, countries_by_continents)
    global_data['Financial flows to developing countries (US $)'] = global_data['Financial flows to developing countries (US $)'].div(
        1e9)

    renewable_energy_in_final_consumption = global_data[[
        'Country', 'Year', 'Renewable energy share in the total final energy consumption (%)', 'Continent']]
    renewable_energy_in_final_consumption = renewable_energy_in_final_consumption.dropna()

    renewable_energy_in_final_consumption_per_continent = (renewable_energy_in_final_consumption
                                                           .groupby('Continent')
                                                           ['Renewable energy share in the total final energy consumption (%)']
                                                           .agg(['mean', 'median'])
                                                           .round(2)
                                                           .sort_values(by='mean', ascending=True)
                                                           .reset_index())

    fig = px.bar(renewable_energy_in_final_consumption_per_continent, orientation='h', x='mean', y='Continent',
                 color='Continent', color_discrete_sequence=px.colors.qualitative.Prism,
                 title='Consumo de energia provindo de fontes renováveis')
    fig.update_layout(xaxis_title='Percentual do total consumido (%)',
                      yaxis_title='Continente', showlegend=False)
    st.plotly_chart(fig)

    # Subseção 2.2: Taxa de uso de cada fonte não fóssil
    st.header("Taxa de uso de cada fonte de energia não fóssil")

    # Carregar os dados necessários
    hydro_energy_participation_per_country = data['hydro_energy_participation_per_country']
    wind_energy_participation_per_country = data['wind_energy_participation_per_country']
    solar_energy_participation_per_country = data['solar_energy_participation_per_country']

    # Mapa: Fontes hidrelétricas
    hydro_energy_participation_per_country = hydro_energy_participation_per_country.dropna().round(2)
    hydro_energy_participation_per_country.sort_values(
        by='Year', ascending=True, inplace=True)
    hydro_energy_participation_per_country = hydro_energy_participation_per_country.query(
        'Year >= 2000 and Year <= 2021')
    plot_animated_choropleth_map(hydro_energy_participation_per_country, 'Entity',
                                 'Hydro (% electricity)',
                                 'Taxa de participação de energias hidrelétricas nas matrizes mundiais (2000-2021)',
                                 'Porcentagem (%)')

    # Gráfico de linhas: Fontes hidrelétricas (Média Global)
    global_hydro_energy_participation = hydro_energy_participation_per_country.query(
        'Entity == "World" and Year >= 2000 and Year <= 2021')[['Year', 'Hydro (% electricity)']]
    plot_line_chart_energy_source_use(global_hydro_energy_participation, 'Year', 'Hydro (% electricity)',
                                      'Taxa média global de participação de energias hidrelétricas nas matrizes mundiais(2000-2021)',
                                      'Ano', 'Porcentagem (%)')

    # Mapa: Fontes solares
    solar_energy_participation_per_country = solar_energy_participation_per_country.dropna().round(2)
    solar_energy_participation_per_country.sort_values(
        by='Year', ascending=True, inplace=True)
    solar_energy_participation_per_country = solar_energy_participation_per_country.query(
        'Year >= 2000 and Year <= 2021')
    plot_animated_choropleth_map(solar_energy_participation_per_country, 'Entity',
                                 'Solar (% electricity)',
                                 'Taxa de participação de energia solar nas matrizes mundiais (2000-2021)',
                                 'Porcentagem (%)')

    # Gráfico de linhas: Fontes solares (Média Global)
    global_solar_energy_participation = solar_energy_participation_per_country.query(
        'Entity == "World" and Year >= 2000 and Year <= 2021')[['Year', 'Solar (% electricity)']]
    plot_line_chart_energy_source_use(global_solar_energy_participation, 'Year', 'Solar (% electricity)',
                                      'Taxa média global de participação de energia solar nas matrizes mundiais (2000-2021)',
                                      'Ano', 'Porcentagem (%)')

    # Mapa: Fontes eólicas
    wind_energy_participation_per_country = wind_energy_participation_per_country.dropna().round(2)
    wind_energy_participation_per_country.sort_values(
        by='Year', ascending=True, inplace=True)
    wind_energy_participation_per_country = wind_energy_participation_per_country.query(
        'Year >= 2000 and Year <= 2021')
    plot_animated_choropleth_map(wind_energy_participation_per_country, 'Entity',
                                 'Wind (% electricity)',
                                 'Taxa de participação de energia eólica nas matrizes mundiais (2000-2021)',
                                 'Porcentagem (%)')

    # Gráfico de linhas: Fontes eólicas (Média Global)
    global_wind_energy_participation = wind_energy_participation_per_country.query(
        'Entity == "World" and Year >= 2000 and Year <= 2021')[['Year', 'Wind (% electricity)']]
    plot_line_chart_energy_source_use(global_wind_energy_participation, 'Year', 'Wind (% electricity)',
                                      'Taxa média global de participação de energias eólicas nas matrizes mundiais (2000-2021)',
                                      'Ano', 'Porcentagem (%)')


# Seção 3: Relação entre Riqueza e Energias Renováveis
def wealth_renewable_relationship():
    st.title("Relação entre Riqueza e Energias Renováveis")

    # Subseção 3.1: PIB per capita vs. Uso de energia renovável
    st.header("PIB per capita vs. Uso de energia renovável")
    df = data['data']
    media_paises = (df[['Entity', 'Year', 'Renewable energy share in the total final energy consumption (%)', 'gdp_per_capita']].
                    dropna().
                    groupby('Entity').
                    mean().
                    reset_index()
                    )
    fig = px.scatter(
        data_frame=media_paises,
        x='gdp_per_capita',
        y='Renewable energy share in the total final energy consumption (%)',
        title='PIB per capita X Uso de energia renovável',
        labels={'gdp_per_capita': 'PIB Per Capita (U$)',
                'Renewable energy share in the total final energy consumption (%)': 'Uso de energia renovável (%)'},
        hover_name='Entity'
    )
    st.plotly_chart(fig)

    st.markdown("No gráfico, cada ponto significa um país, considerando a média tanto do PIB Per Capita quanto do uso de energias renováveis. Podemos notar que o PIB Per Capita não parece estar diretamente relacionado com o uso dessas energias, como mostra a grande variação, principalmente, entre países mais pobres.")

    # Subseção 3.2: PIB per capita vs. Capacidade de geração de energia limpa
    st.header("PIB per capita vs. Capacidade de geração de energia limpa")
    df_sorted = (df[['Entity', 'Renewable-electricity-generating-capacity-per-capita', 'Primary energy consumption per capita (kWh/person)', 'gdp_per_capita']].
                 dropna().
                 groupby('Entity').
                 mean().
                 reset_index().
                 sort_values(by='gdp_per_capita', ascending=False).
                 reset_index(drop=True)
                 )
    fig = px.scatter(
        data_frame=df_sorted,
        x='gdp_per_capita',
        y='Renewable-electricity-generating-capacity-per-capita',
        color_discrete_sequence=['blue'],
        title='PIB per capita X Capacidade de geração de energia limpa por pessoa',
        labels={'Renewable-electricity-generating-capacity-per-capita': 'Capacidade de geração de energia limpa per capita',
                'gdp_per_capita': 'PIB per capita (U$)'},
        hover_name='Entity'
    )
    st.plotly_chart(fig)

    st.markdown("Novamente, como pode-se observar, não parece haver uma relação muito clara entre PIB per capita e a capacidade dos países de gerar energia de fontes renováveis. A maioria dos países, considerando apenas o eixo Y, estão próximos uns dos outros.")

    # Subseção 3.3: Distribuição do uso de energia per capita
    st.header("Distribuição do uso de energia per capita")
    df_sorted = df_sorted.assign(Classificacao='Pobres')
    df_sorted.loc[:int(len(df_sorted)/2), 'Classificacao'] = 'Ricos'
    fig = px.box(
        data_frame=df_sorted,
        x='Classificacao',
        y='Primary energy consumption per capita (kWh/person)',
        color='Classificacao',
        title='Distribuição do uso de energia per capita entre países ricos e pobres',
        labels={'Primary energy consumption per capita (kWh/person)': 'Uso de energia (kWh/pessoa)',
                'Classificacao': 'Classificação'}
    )
    st.plotly_chart(fig)
    st.markdown("Conforme pode ser visto, a divisão entre países ricos e pobres (metade dos dados para cada lado) mostra que países com PIB per capita acima da mediana possuem uma variação de consumo muito maior, tendo, por exemplo, a mediana próxima ao máximo dos países mais pobres. Logo, parece haver alguma relação entre PIB per capita e consumo de energia per capita.")

    # Subseção 3.4: Análise de países específicos (Brasil, EUA, China, Luxemburgo)
    st.header("Análise de países específicos")
    
    st.markdown("A efeito de comparação com o Brasil em relação ao PIB per Capita e porcentagem de participação de energias renováveis, foram selecionados os países: Estados Unidos, China e Luxemburgo. ")
    
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'PIB per Capita (U$)', '% da participação da energias renováveis'],
        row_titles=['Brasil', 'EUA', 'China', 'Luxemburgo']
    )

    index = {1: 'Brazil', 2: 'United States', 3: 'China', 4: 'Luxembourg'}
    colors = {1: '#00ad00', 2: '#1f77b4', 3: '#ff0000', 4: '#3182EB'}

    for i in index:
        df1 = df[df['Entity'] == index[i]].dropna(
            subset=['Year', 'Renewable energy share in the total final energy consumption (%)', 'gdp_per_capita'])
        fig1 = px.line(data_frame=df1, x='Year',
                       y='Renewable energy share in the total final energy consumption (%)')
        fig2 = px.line(data_frame=df1, x='Year', y='gdp_per_capita')
        fig1.update_traces(line_color=colors[i])
        fig2.update_traces(line_color=colors[i])
        for trace in fig1.data:
            fig.add_trace(trace, row=i, col=2)
        for trace in fig2.data:
            fig.add_trace(trace, row=i, col=1)

    fig.update_layout(
        autosize=False,
        width=1000,
        height=800,
        margin=dict(r=10)
    )

    st.plotly_chart(fig)
    st.markdown("Dentre os países selecionados, alguns parecem apresentar uma certa relação entre o crescimento do PIB per capita e o uso de energias renováveis, como o Brasil e os EUA. Entretanto, podemos notar que na China, aparentemente, o oposto ocorreu. Em Luxemburgo, a participação dessas energias aumentou, mas permanece baixa (ainda mais considerando o seu alto PIB per capita).")


# Seção 4: Regressão linear e previsões
def linear_regression_prevision():
    brazil_data = data[data['Entity'] == 'Brazil']

    brazil_data = brazil_data[['Year', 'Electricity from fossil fuels (TWh)', 'Electricity from renewables (TWh)']].dropna()

    X = brazil_data['Year'].values.reshape(-1, 1)
    y_fossil = brazil_data['Electricity from fossil fuels (TWh)'].values
    y_renewables = brazil_data['Electricity from renewables (TWh)'].values

    model_fossil = LinearRegression().fit(X, y_fossil)
    model_renewables = LinearRegression().fit(X, y_renewables)

    future_years = np.array([2023 + i for i in range(10)]).reshape(-1, 1)
    all_years = np.append(X, future_years).reshape(-1, 1)
    fossil_pred_all = model_fossil.predict(all_years)
    renewables_pred_all = model_renewables.predict(all_years)

    # Calcular os IC 95%
    def calculate_confidence_interval(model, X, y, X_all, confidence=0.95):
        y_pred = model.predict(X)
        residual = y - y_pred
        mean_x = np.mean(X)
        n = len(X)
        t = 1.96  # valor t para a confiança de 95% e graus de liberdade grandes
        s_err = np.sqrt(np.sum(residual**2) / (n - 2))
        interval = t * s_err * np.sqrt(1/n + (X_all.flatten() - mean_x)**2 / np.sum((X - mean_x)**2))
        return interval.flatten()

    # IC para combustíveis fósseis
    fossil_intervals_all = calculate_confidence_interval(model_fossil, X, y_fossil, all_years)

    # IC para renováveis
    renewables_intervals_all = calculate_confidence_interval(model_renewables, X, y_renewables, all_years)

    # Obter limites do gráfico para manter as escalas semelhantes
    min_fossil = np.min(np.concatenate((y_fossil, fossil_pred_all - fossil_intervals_all)))
    max_fossil = np.max(np.concatenate((y_fossil, fossil_pred_all + fossil_intervals_all)))
    min_renewables = np.min(np.concatenate((y_renewables, renewables_pred_all - renewables_intervals_all)))
    max_renewables = np.max(np.concatenate((y_renewables, renewables_pred_all + renewables_intervals_all)))

    overall_min = min(min_fossil, min_renewables)
    overall_max = max(max_fossil, max_renewables)

    # Plotar as previsões
        
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Previsão da Produção de Energia de Combustíveis Fósseis no Brasil',
                                        'Previsão da Produção de Energia Renovável no Brasil'))

    fig.add_trace(go.Scatter(x=X, y=y_fossil, mode='markers', name='Dados reais', marker=dict(color='blue')),
                row=1, col=1)
    fig.add_trace(go.Scatter(x=all_years, y=fossil_pred_all, mode='lines', name='Previsões', line=dict(color='red')),
                row=1, col=1)
    fig.add_trace(go.Scatter(x=all_years + all_years[::-1], 
                            y=list(fossil_pred_all - fossil_intervals_all) + list(fossil_pred_all + fossil_intervals_all)[::-1], 
                            fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', line=dict(color='rgba(255, 0, 0, 0)'), 
                            name='Intervalo de Confiança de 95%'),
                row=1, col=1)
    
    fig.add_trace(go.Scatter(x=X, y=y_renewables, mode='markers', name='Dados reais', marker=dict(color='green')),
                row=1, col=2)
    fig.add_trace(go.Scatter(x=all_years, y=renewables_pred_all, mode='lines', name='Previsões', line=dict(color='orange')),
                row=1, col=2)
    fig.add_trace(go.Scatter(x=all_years + all_years[::-1], 
                            y=list(renewables_pred_all - renewables_intervals_all) + list(renewables_pred_all + renewables_intervals_all)[::-1], 
                            fill='toself', fillcolor='rgba(255, 165, 0, 0.2)', line=dict(color='rgba(255, 165, 0, 0)'), 
                            name='Intervalo de Confiança de 95%'),
                row=1, col=2)

    fig.update_layout(
        height=600, 
        width=1200,
        showlegend=True,
        xaxis_title='Ano',
        yaxis_title='Produção de Energia de Combustíveis Fósseis (TWh)',
        xaxis2_title='Ano',
        yaxis2_title='Produção de Energia Renovável (TWh)',
        xaxis=dict(range=[min(all_years), max(all_years)]),
        yaxis=dict(range=[overall_min, overall_max]),
        xaxis2=dict(range=[min(all_years), max(all_years)]),
        yaxis2=dict(range=[overall_min, overall_max]),
        title_text='Previsões de Produção de Energia no Brasil'
    )
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y_fossil, mode='markers', name='Dados reais', marker=dict(color='blue')))
    st.plotly_chart(fig)
    

# Barra lateral para navegação
st.sidebar.title("Navegação")
options = st.sidebar.radio("Selecione uma seção:", [
    "Análise Global",
    "Consumo por Localidade",
    "Relação entre Riqueza e Energias Renováveis",
    "Regressão Linear e previsões"
])

# Exibir a seção selecionada
if options == "Análise Global":
    analyze_global_data()
elif options == "Consumo por Localidade":
    consumption_by_location()
elif options == "Relação entre Riqueza e Energias Renováveis":
    wealth_renewable_relationship()
elif options == "Regressão Linear e previsões":
    linear_regression_prevision()
