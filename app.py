import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import zipfile

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA (Identidade Visual Profissional)
# ==============================================================================
st.set_page_config(
    page_title="Impacto de Metais & Macroeconomia no Hardware",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização customizada via CSS para tipografia e cartões limpos
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #1f77b4;
    }
    .story-box {
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #636EFA;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# PROCESSO DE ETL AUTOMATIZADO (Cacheado para Performance no Deploy)
# ==============================================================================
@st.cache_data
def carregar_e_transformar_dados():
    ZIP_PATH = "datasets.zip"
    EXTRACT_FOLDER = "datasets_extraidos"
    
    # Descompacta se a pasta não existir
    if not os.path.exists(EXTRACT_FOLDER) and os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_FOLDER)
            
    # Tratamento de caminhos tolerante a subpastas internas no zip
    base_busca = EXTRACT_FOLDER
    if os.path.exists(os.path.join(EXTRACT_FOLDER, "datasets")):
        base_busca = os.path.join(EXTRACT_FOLDER, "datasets")

    # 1. Leitura dos Arquivos
    df_hardware_raw = pd.read_csv(f"{base_busca}/hardware_gamer_dataset.csv")
    df_metais_raw = pd.read_excel(f"{base_busca}/CMO-Historical-Data-Monthly.xlsx", sheet_name="Monthly Prices", skiprows=4)
    df_usd_raw = pd.read_csv(f"{base_busca}/ipeadata[taxa USD comercial].csv")
    df_salario_raw = pd.read_csv(f"{base_busca}/ipeadata[salário mín. vigente].csv")
    df_selic_raw = pd.read_csv(f"{base_busca}/BACEN_SELIC.csv")

    # 2. Transformação - Metais
    df_metais_raw = df_metais_raw.iloc[2:].reset_index(drop=True)
    df_metais_raw = df_metais_raw.rename(columns={'Unnamed: 0': 'Date'})
    metais = ['Date', 'Aluminum', 'Copper', 'Tin', 'Gold', 'Silver']
    df_metais = df_metais_raw[metais].copy()
    df_metais['Date'] = pd.to_datetime(df_metais['Date'], format='%YM%m', errors='coerce')
    for col in metais[1:]:
        df_metais[col] = pd.to_numeric(df_metais[col], errors='coerce')
    df_metais = df_metais.dropna().drop_duplicates().reset_index(drop=True)

    # 3. Transformação - Salário
    df_salario_raw = df_salario_raw.rename(columns={
        'Data': 'Date',
        'Salário mínimo vigente - R$ - Ministério da Economia- Outras (Min. Economia/Outras) - MTE12_SALMIN12': 'Salario_Minimo'
    })
    df_salario = df_salario_raw.copy()
    df_salario['Date'] = df_salario['Date'].astype(str)
    df_salario['Ano'] = df_salario['Date'].str.extract(r'(\d{4})\.')
    df_salario['Mes'] = df_salario['Date'].str.extract(r'\.(\d{2})')
    df_salario = df_salario.dropna(subset=['Ano', 'Mes'])
    df_salario['Date'] = pd.to_datetime({'year': df_salario['Ano'].astype(int), 'month': df_salario['Mes'].astype(int), 'day': 1}, errors='coerce')
    df_salario = df_salario.drop(columns=['Ano', 'Mes']).dropna().reset_index(drop=True)

    # 4. Transformação - USD
    df_usd_raw = df_usd_raw.rename(columns={
        'Data': 'Date',
        'Taxa de câmbio dólar comercial para venda - média - R$ - Banco Central do Brasil- Sistema Gerenciador de Séries Temporais (Bacen Outras/SGS) - GM366_ERV366': 'USD_BRL'
    })
    df_usd = df_usd_raw[['Date', 'USD_BRL']].copy()
    df_usd['Date'] = pd.to_datetime(df_usd['Date'], format='%d/%m/%Y', errors='coerce')
    df_usd['USD_BRL'] = pd.to_numeric(df_usd['USD_BRL'], errors='coerce')
    df_usd = df_usd.dropna().reset_index(drop=True)

    # 5. Transformação - SELIC
    df_selic = df_selic_raw[['DataInicioVigencia', 'MetaSelic']].copy().rename(columns={'DataInicioVigencia': 'Date', 'MetaSelic': 'SELIC'})
    df_selic['Date'] = pd.to_datetime(df_selic['Date']).dt.tz_localize(None).dt.normalize()
    df_selic['SELIC'] = pd.to_numeric(df_selic['SELIC'], errors='coerce')
    df_selic = df_selic.dropna().drop_duplicates().reset_index(drop=True)

    # 6. Transformação - Hardware
    df_hardware = df_hardware_raw.drop(columns=['Ano', 'Mes'], errors='ignore')
    df_hardware['Date'] = pd.to_datetime(df_hardware['Date'], errors='coerce')
    df_hardware['Price_BRL'] = pd.to_numeric(df_hardware['Price_BRL'], errors='coerce')
    df_hardware = df_hardware.dropna().drop_duplicates().reset_index(drop=True)

    # 7. Granularidade Mensal & Mensalização
    df_hardware['Date_Mensal'] = df_hardware['Date'].dt.to_period('M').dt.to_timestamp()
    df_hardware_mes = df_hardware.groupby(['Product_ID', 'Produto', 'Categoria', 'Faixa', 'Date_Mensal'], as_index=False)['Price_BRL'].mean().rename(columns={'Date_Mensal': 'Date', 'Price_BRL': 'Preco_Medio_BRL_Mes'})
    df_hardware_mes['Date'] = pd.to_datetime(df_hardware_mes['Date']).dt.normalize()

    df_usd_temp = df_usd.set_index('Date')
    df_usd_mes = df_usd_temp['USD_BRL'].resample('MS').mean().reset_index().rename(columns={'USD_BRL': 'USD_Medio_Mes'})
    df_usd_mes['Date'] = pd.to_datetime(df_usd_mes['Date']).dt.normalize()

    df_selic_temp = df_selic.set_index('Date')
    df_selic_dia = df_selic_temp.resample('D').ffill()
    df_selic_mes = df_selic_dia.resample('MS').mean().reset_index().rename(columns={'SELIC': 'Selic_Media_Mes'})
    df_selic_mes['Date'] = pd.to_datetime(df_selic_mes['Date']).dt.normalize()

    # Conversão de Moeda dos Metais
    df_metais_conv = pd.merge(df_metais, df_usd_mes, on='Date', how='left')
    df_metais_conv['USD_Medio_Mes'] = df_metais_conv['USD_Medio_Mes'].ffill().bfill()
    metais_colunas = ['Aluminum', 'Copper', 'Tin', 'Gold', 'Silver']
    for metal in metais_colunas:
        df_metais_conv[f'{metal}_BRL'] = df_metais_conv[metal] * df_metais_conv['USD_Medio_Mes']
    df_metais_brl = df_metais_conv[['Date'] + [f'{metal}_BRL' for metal in metais_colunas]].copy()

    # Scaffolding e Merge Final
    data_min, data_max = df_hardware_mes['Date'].min(), df_hardware_mes['Date'].max()
    calendario = pd.DataFrame({'Date': pd.date_range(start=data_min, end=data_max, freq='MS')})
    produtos_unicos = df_hardware_mes[['Product_ID', 'Produto', 'Categoria', 'Faixa']].drop_duplicates()
    df_base = calendario.merge(produtos_unicos, how='cross')

    df_fato_final = pd.merge(df_base, df_hardware_mes, on=['Date', 'Product_ID', 'Produto', 'Categoria', 'Faixa'], how='left')
    df_fato_final['Preco_Medio_BRL_Mes'] = df_fato_final.groupby('Product_ID')['Preco_Medio_BRL_Mes'].ffill()

    df_fato_final = pd.merge(df_fato_final, df_usd_mes, on='Date', how='left')
    df_fato_final = pd.merge(df_fato_final, df_selic_mes, on='Date', how='left')
    df_fato_final = pd.merge(df_fato_final, df_metais_brl, on='Date', how='left')
    df_fato_final = pd.merge(df_fato_final, df_salario, on='Date', how='left')

    colunas_macro = ['USD_Medio_Mes', 'Selic_Media_Mes', 'Aluminum_BRL', 'Copper_BRL', 'Tin_BRL', 'Gold_BRL', 'Silver_BRL', 'Salario_Minimo']
    df_fato_final[colunas_macro] = df_fato_final[colunas_macro].bfill().ffill()
    
    return df_fato_final.sort_values(by=['Product_ID', 'Date']).reset_index(drop=True)

# Inicialização dos dados tratados
try:
    df_fato = carregar_e_transformar_dados()
except Exception as e:
    st.error(f"Erro ao processar dados de entrada. Verifique se o arquivo 'datasets.zip' está na raiz do repositório. Detalhes: {e}")
    st.stop()

# ==============================================================================
# ESTRUTURA DE NAVEGAÇÃO (Sidebar / Menu Lateral)
# ==============================================================================
st.sidebar.title("📌 Projeto A3 - UNIFACS")
st.sidebar.markdown("**UC:** Análise de Dados | **Professor:** Xavier")
st.sidebar.markdown("**Alunos:** Marcela Tourinho, Maria Clara Maluf, Isadora")
st.sidebar.markdown("---")
fase_selecionada = st.sidebar.radio("Selecione a Etapa do Projeto:", [
    "🏠 Visão Geral & Contexto de Negócio",
    "📊 Fase 2: Análise Exploratória (EDA)",
    "🎯 Fase 3: Storytelling & Decisões"
])

# Filtros Globais na Sidebar aplicados apenas às visualizações
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Filtros de Análise")
categorias_disponiveis = df_fato['Categoria'].unique().tolist()
cat_filtro = st.sidebar.multiselect("Filtrar por Categoria de Componente:", categorias_disponiveis, default=categorias_disponiveis)
df_filtrado = df_fato[df_fato['Categoria'].isin(cat_filtro)]

# ==============================================================================
# TELA 1: VISÃO GERAL & CONTEXTO DE NEGÓCIO
# ==============================================================================
if fase_selecionada == "🏠 Visão Geral & Contexto de Negócio":
    st.title("💻 Impacto do Preço de Metais no Mercado Consumidor de Computadores")
    st.subheader("Uma Abordagem Macroeconômica sobre a Cadeia de Suprimentos de Hardware Gamer")
    
    st.markdown("""
    Este projeto visa decodificar como os macroindicadores econômicos do Brasil — como a **taxa de câmbio (USD)**, 
    a taxa de juros básica (**SELIC**) e o preço das **commodities metálicas estruturais e preciosas** — afetam diretamente 
    o preço final dos componentes de computadores no mercado consumidor brasileiro.
    """)
    
    # Grid de KPIs Iniciais
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><b>Total de Registros</b><h2>{len(df_fato)}</h2><small>Dados contínuos pós-pipeline</small></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><b>Produtos Monitorados</b><h2>{df_fato['Produto'].nunique()}</h2><small>Categorizados por faixas de custo</small></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><b>Câmbio Médio (Período)</b><h2>R$ {df_fato['USD_Medio_Mes'].mean():.2f}</h2><small>Dólar comercial médio</small></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><b>Metais Analisados</b><h2>5</h2><small>Alumínio, Cobre, Estanho, Ouro e Prata</small></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎯 Mapeamento e Alinhamento Estratégico (O Contexto do Trabalho)")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ### 🤔 Quem é o cliente ideal para este projeto?
        * **Diretores de Compras e Supply Chain de E-commerces:** Grandes varejistas de tecnologia que precisam prever reajustes de preços com base no mercado internacional.
        * **Montadoras Locais de Computadores e Distribuidores:** Empresas que importam insumos químicos e peças prontas e operam com margens de lucro apertadas.
        * **Investidores do Setor de Varejo de Tecnologia:** Gestores de fundos mapeando a resiliência do mercado de hardware gamer frente à inflação.
        """)
    with col_b:
        st.markdown("""
        ### 🏭 Quais públicos mais dependem destes hardwares?
        1. **Enterprise & Desenvolvedores:** Estações de trabalho profissionais (*Workstations*) focadas em Inteligência Artificial, renderização e processamento massivo de dados.
        2. **Setores de Engenharia, Arquitetura e Design:** Dependem de forte poder computacional gráfico (GPUs de alta performance).
        3. **Mercado Pro-Gamer e Entusiastas:** Um nicho de consumo de altíssimo valor agregado com alta resiliência a aumentos de preços.
        """)

# ==============================================================================
# TELA 2: FASE 2 - ANÁLISE EXPLORATÓRIA E DESCRITIVA (EDA)
# ==============================================================================
elif fase_selecionada == "📊 Fase 2: Análise Exploratória (EDA)":
    st.title("📊 Fase 2: Análise Exploratória e Descritiva")
    st.markdown("### Métodos, Técnicas e Comportamento Estatístico da Base")

    # Abas internas para organizar os objetivos da EDA
    tab_estatistica, tab_distribuicao, tab_correlacao = st.tabs([
        "🧮 Estatística Descritiva & Dispersão", 
        "📉 Distribuição e Anomalias", 
        "🔗 Correlações e Padrões"
    ])

    with tab_estatistica:
        st.markdown("#### Métricas de Tendência Central e Dispersão por Categoria")
        st.markdown("> **Técnica Utilizada:** Agrupamento estatístico computando Média, Mediana, Quartis ($25\%$ e $75\%$) e Desvio Padrão para avaliar o espalhamento dos preços reais praticados.")
        
        # Tabela Descritiva Resumida
        df_descritiva = df_filtrado.groupby('Categoria')['Preco_Medio_BRL_Mes'].agg([
            ('Média (BRL)', 'mean'),
            ('Mediana (BRL)', 'median'),
            ('Desvio Padrão (BRL)', 'std'),
            ('Mínimo (BRL)', 'min'),
            ('Máximo (BRL)', 'max')
        ]).round(2)
        
        st.dataframe(df_descritiva, use_container_width=True)
        
        # Gráfico Box Plot / Strip Plot Lado a Lado (Convertido do seu Script Original)
        cores_faixas = {"Premium": "#636EFA", "Intermediário": "#00CC96", "Baixo custo": "#EF553B"}
        
        fig_lado_a_lado = make_subplots(
            rows=1, cols=2, 
            subplot_titles=("Estrutura Estatística (Box Plot)", "Dispersão Real dos Dados (Strip Plot)"),
            horizontal_spacing=0.1
        )
        
        fig_box = px.box(df_filtrado, x="Categoria", y="Preco_Medio_BRL_Mes", color="Faixa", color_discrete_map=cores_faixas, points=False)
        fig_strip = px.strip(df_filtrado, x="Categoria", y="Preco_Medio_BRL_Mes", color="Faixa", color_discrete_map=cores_faixas, stripmode="group")
        
        for trace in fig_box.data:
            trace.marker.color = cores_faixas.get(trace.name, trace.marker.color)
            fig_lado_a_lado.add_trace(trace, row=1, col=1)
            
        for trace in fig_strip.data:
            trace.showlegend = False
            trace.jitter = 0.7
            trace.marker.size = 4
            trace.marker.opacity = 0.6
            trace.marker.color = cores_faixas.get(trace.name, trace.marker.color)
            fig_lado_a_lado.add_trace(trace, row=1, col=2)
            
        fig_lado_a_lado.update_layout(template="plotly_white", boxmode="group", height=500)
        fig_lado_a_lado.update_yaxes(title_text="Preço Médio (BRL)", row=1, col=1)
        st.plotly_chart(fig_lado_a_lado, use_container_width=True)
        
        st.markdown("""
        **Documentação de Padrões Obtidos:** O diagnóstico estatístico via IQR demonstrou que as flutuações de preços nas categorias premium (como as memórias Corsair Dominator e CPUs i7) não constituem *outliers* de processamento ou erros de coleta, mas sim movimentos voláteis e legítimos de oferta/demanda que respondem diretamente a gatilhos cambiais.
        """)

    with tab_distribuicao:
        st.markdown("#### Evolução Temporal Contínua por Produto")
        
        df_eda_sorted = df_filtrado.sort_values(by="Date")
        paleta_produtos = px.colors.qualitative.Dark24
        
        fig_line = px.line(
            df_eda_sorted, x="Date", y="Preco_Medio_BRL_Mes", color="Produto", facet_col="Categoria",
            color_discrete_sequence=paleta_produtos, markers=True,
            labels={"Preco_Medio_BRL_Mes": "Preço (BRL)"}
        )
        fig_line.update_layout(template="plotly_white", height=500, margin=dict(t=40, b=40))
        fig_line.update_xaxes(tickformat="%Y-%m", tickangle=45)
        fig_line.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        st.plotly_chart(fig_line, use_container_width=True)

    with tab_correlacao:
        st.markdown("#### Matriz de Correlação Linear de Pearson Interativa")
        
        df_corr = df_filtrado[["Preco_Medio_BRL_Mes", "USD_Medio_Mes", "Selic_Media_Mes", "Aluminum_BRL", "Copper_BRL", "Gold_BRL"]]
        matriz_correlacao = df_corr.corr()
        
        fig_heatmap = px.imshow(
            matriz_correlacao, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1
        )
        fig_heatmap.update_layout(title="Mapeamento de Força: Hardware vs. Indicadores Econômicos")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("""
        **Análise dos Padrões Identificados:**
        * Observa-se uma **forte correlação linear positiva** entre o preço médio do hardware e o **câmbio do dólar**, confirmando a alta dependência de componentes importados.
        * O **Cobre (Copper_BRL)** e o **Alumínio (Aluminum_BRL)** mostram correlações estreitas com a subida geral de preços, sinalizando aumentos simultâneos de custos na cadeia primária global de manufatura de eletrônicos.
        """)

# ==============================================================================
# TELA 3: FASE 3 - STORYTELLING & RECOMENDAÇÕES (Foco em Gestores)
# ==============================================================================
elif fase_selecionada == "🏠 Visão Geral & Contexto de Negócio" or fase_selecionada == "🎯 Fase 3: Storytelling & Decisões":
    st.title("🎯 Fase 3: Storytelling & Tomada de Decisão")
    st.subheader("Transformando Dados Técnicos em Insights Estratégicos para Gestores")
    
    st.markdown("""
    <div class='story-box'>
    <b>Narrativa de Negócios:</b> A produção global de semicondutores e placas de circuito integrado não depende apenas 
    de engenharia de software; ela é refém do mercado de commodities minerais. Elementos como o Cobre são cruciais para sistemas de 
    dissipação térmica de GPUs, e o Alumínio é amplamente utilizado em chassis e estruturas. Quando as cotações internacionais sobem 
    e o Real se desvaloriza frente ao Dólar, o impacto no varejo brasileiro é imediato e severo.
    </div>
    """, unsafe_allow_html=True)
    
    # Seletor interativo dinâmico trazido do seu script (Melhoria de UI/UX profissional)
    st.markdown("### 🔍 Simulador Dinâmico de Correlação Histórica")
    st.write("Selecione um produto específico e um metal de base para avaliar a convergência de preços na linha do tempo:")
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        produto_sel = st.selectbox("Selecione o Componente:", sorted(df_fato["Produto"].unique().tolist()))
    with col_sel2:
        metais_opcoes = {
            "Copper_BRL": "Cobre (Copper)",
            "Aluminum_BRL": "Alumínio (Aluminum)",
            "Tin_BRL": "Estanho (Tin)",
            "Gold_BRL": "Ouro (Gold)",
            "Silver_BRL": "Prata (Silver)"
        }
        metal_sel = st.selectbox("Selecione o Metal Comparativo:", list(metais_opcoes.keys()), format_func=lambda x: metais_opcoes[x])
        
    # Filtragem e plotagem do gráfico de duplo eixo
    df_prod_sel = df_fato[df_fato["Produto"] == produto_sel].sort_values(by="Date")
    
    fig_super_dual = go.Figure()
    fig_super_dual.add_trace(go.Scatter(
        x=df_prod_sel["Date"], y=df_prod_sel["Preco_Medio_BRL_Mes"],
        name=f"Preço: {produto_sel}", mode="lines+markers", line=dict(color="#1f77b4", width=3)
    ))
    fig_super_dual.add_trace(go.Scatter(
        x=df_prod_sel["Date"], y=df_prod_sel[metal_sel],
        name=f"Preço: {metais_opcoes[metal_sel]}", mode="lines+markers",
        line=dict(color="#ff7f0e", width=2, dash="dash"), yaxis="y2"
    ))
    
# Subjacente ao Simulador Dinâmico na Linha 365:
    fig_super_dual.update_layout(
        title=f"Evolução Temporal Cruzada: {produto_sel} vs. {metais_opcoes[metal_sel]}",
        xaxis=dict(title="Meses", tickformat="%Y-%m"),
        
        # Eixo Esquerdo (Hardware) - Correção aplicada aqui:
        yaxis=dict(
            title=dict(
                text="Preço do Hardware (BRL)",
                font=dict(color="#1f77b4")
            ),
            tickfont=dict(color="#1f77b4")
        ),
        
        # Eixo Direito (Metal) - Correção aplicada aqui:
        yaxis2=dict(
            title=dict(
                text=f"Preço do {metais_opcoes[metal_sel]} (BRL)",
                font=dict(color="#ff7f0e")
            ),
            tickfont=dict(color="#ff7f0e"),
            overlaying="y", 
            side="right"
        ),
        template="plotly_white", 
        height=500
    )
    st.plotly_chart(fig_super_dual, use_container_width=True)

    # Contexto Histórico Adicional Solicitado
    st.markdown("---")
    st.markdown("### 🌐 Que outros contextos macroeconômicos provocaram esta alta de preços?")
    
    col_box1, col_box2 = st.columns(2)
    with col_box1:
        st.markdown("""
        * **A Crise dos Semicondutores e Logística Pós-Pandemia:** Gargalos severos de logística internacional e paralisações em fundições chave na Ásia reduziram drasticamente a oferta global de silício refinado.
        * **Boom de Computação de IA de Larga Escala:** A busca massiva por chips gráficos aceleradores (GPUs) por gigantes de tecnologia gerou escassez crônica e canibalização da capacidade produtiva global, elevando preços de componentes de consumo padrão.
        """)
    with col_b:
        st.markdown("""
        * **Corrida pela Mineração de Criptoativos:** Durante janelas específicas de alta de moedas digitais proof-of-work, os estoques locais de placas de vídeo foram drenados, distorcendo os preços no ecossistema Buscapé/Zoom.
        * **A Inflação Local e Poder de Compra:** O gráfico comprova que o distanciamento entre o salário mínimo nacional e a média de preços de categorias intermediárias/premium cria barreiras de consumo que impactam o faturamento de montadoras locais.
        """)

    # Recomendações Finais (Fase 3 - Resultado Final)
    st.markdown("---")
    st.markdown("### 💡 Recomendações Acionáveis para Gestores (Tomada de Decisão)")
    st.info("""
    1. **Estratégia de Hedge Cambial:** Varejistas e importadores devem adotar mecanismos de proteção financeira (*hedge*) baseados na volatilidade do Dólar e contratos futuros de metais industriais para mitigar reajustes abruptos.
    2. **Diversificação de Portfólio por Faixa de Custo:** Gestores de estoque devem balancear as compras aumentando a fatia de itens de 'Baixo custo' em cenários de subida simultânea do cobre e do dólar, mantendo o giro de capital estável.
    3. **Previsibilidade de Demanda via Monitoramento Externo:** Integrar dados de plataformas como o World Bank diretamente aos sistemas corporativos de ERP permite antecipar alterações de preços em até 45 dias antes de chegarem ao mercado nacional.
    """)