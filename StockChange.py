import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import StandardScaler
import requests
from io import BytesIO
import base64

def upload_to_github(file_path, repo, path_in_repo, token):
    with open(file_path, "rb") as f:
        content = base64.b64encode(f.read()).decode()
    url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "message": f"update {path_in_repo}",
        "content": content,
        "branch": "main"
    }
    response = requests.put(url, headers=headers, json=data)
    return response.status_code, response.text

def get_data_from_yfinance():
    import yfinance as yf
    import pandas as pd

    # Liste des tickers IA/Data
    tickers = {
        "Nvidia": "NVDA", "AMD": "AMD", "Intel": "INTC", "Alphabet (Google)": "GOOGL",
        "Microsoft": "MSFT", "Meta": "META", "Amazon": "AMZN", "Apple": "AAPL",
        "Tesla": "TSLA", "Palantir": "PLTR", "C3.ai": "AI", "SoundHound AI": "SOUN",
        "BigBear.ai": "BBAI", "Veritone": "VERI", "Snowflake": "SNOW", "Datadog": "DDOG",
        "Oracle": "ORCL", "IBM": "IBM", "Salesforce": "CRM", "ServiceNow": "NOW",
        "Cisco": "CSCO", "Arista Networks": "ANET", "Elastic NV": "ESTC",
        "Kratos Defense": "KTOS", "Axon Enterprise": "AXON", "Baidu": "BIDU",
        "Alibaba": "BABA", "Tencent": "TCEHY", "SAP": "SAP", "ASML": "ASML",
        "Dassault Systèmes": "DASTY", "Thales": "HO.PA", "Upstart": "UPST",
        "Lemonade": "LMND", "UiPath": "PATH"
    }

    all_data = []
    for name, ticker in tickers.items():
        try:
            data = yf.download(ticker, start="2025-01-01", end=None, interval="1d", progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data.reset_index(inplace=True)
                data["Company"] = name
                data["Ticker"] = ticker
                all_data.append(data)
        except Exception as e:
            print(f"Erreur avec {name}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def train_model():
    st.warning("🚨 Le modèle doit être entraîné depuis la page 'Modélisation / Machine Learning ⚙️'.")

# --- STYLE & SIDEBAR ---

st.sidebar.image("https://raw.githubusercontent.com/AkiraInsight/-STOCKWATCH-/main/Logo%20Akira%20Insight.png", width=150)
# Sidebar Title & Info
st.sidebar.title("📊 StockChange")

# Navigation
pages = [
    "Introduction",
    "Présentation du Dataset",
    "Data Viz 📊",
    "Pré-processing 👷‍♂️",
    "Modélisation / Machine Learning ⚙️",
    "Application 📈"
]
page = st.sidebar.radio("Aller vers", pages)

# Auteurs
st.sidebar.write("__Projet réalisé par :__")
st.sidebar.write("[Tristan Tansu](https://www.linkedin.com/in/tristan-tansu-42009365/)")

# Custom Style
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
    background-color: #f7f7f7 !important;
    }
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    .stApp {
        font-family: 'Orbitron', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #eef3f7 !important;
        color: #333333 !important;
        border-right: 1px solid #d0d7de;
    }
    header {
        background: linear-gradient(180deg, #eef3f7, #dce4eb) !important;
        color: #333333 !important;
        border-bottom: 1px solid #d0d7de;
    }
    .main .block-container {
        background: #ffffff !important;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem auto;
        max-width: 1200px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    div.stButton > button, div.stForm > button {
        transition: transform 0.3s ease, background 0.3s ease, box-shadow 0.3s ease;
        background: linear-gradient(180deg, #eef3f7, #dce4eb);
        color: #333333;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.4rem;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        font-weight: 600;
    }
    div.stButton > button:hover, div.stForm > button:hover {
        transform: scale(1.05);
        background: linear-gradient(180deg, #dce4eb, #bcc4cd);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        color: #333333;
    }
    a {
        color: #0077b6;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    /* Réduction de la largeur de la sidebar */
    [data-testid="stSidebar"] {
        min-width: 300px !important;
        max-width: 400px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if page == pages[0]:
    st.markdown("## 🎯 Introduction – StockChange")
    intro_image_path = "https://raw.githubusercontent.com/AkiraInsight/-STOCKWATCH-/main/Introduction.jpg"
    st.image(intro_image_path, width=800)

    st.markdown("""
        <div style='text-align: justify; font-size: 16px;'>
        Bienvenue dans <b>StockChange</b>, une application interactive d’analyse boursière spécialisée dans les entreprises de l’intelligence artificielle.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🧠 Objectifs de l’application")
    st.markdown("""
    - Suivre l’évolution des cours boursiers des entreprises majeures dans l’IA  
    - Identifier les tendances via des indicateurs techniques (volatilité, momentum…)  
    - Appliquer des modèles de machine learning pour prédire les prix  
    - Aider à la prise de décision d’investissement  
    """)

    st.markdown("---")

    st.markdown("### 👤 Pour qui ?")
    st.markdown("""
        <div style='text-align: justify; font-size: 16px;'>
        Pour les passionnés de tech, de finance, d’IA, ou tout simplement les curieux qui veulent explorer les dynamiques des leaders du marché.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("*Développé par Tristan – Data Analyst & MLOps Engineer @ DataScientest*")

elif page == pages[1]:
    st.markdown("## 📦 Présentation du Dataset")

    dataset_path = "https://raw.githubusercontent.com/AkiraInsight/-STOCKWATCH-/main/stockchange_ai_1y.csv"
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        st.error(f"🚨 Erreur de chargement du dataset : {e}")
        st.stop()

    st.markdown("""
    <div style='text-align: justify; font-size: 16px;'>
    Le dataset utilisé dans ce projet regroupe les données financières quotidiennes de plusieurs entreprises clés du secteur de l’intelligence artificielle, telles que <b>NVIDIA</b>, <b>Alphabet</b>, <b>Microsoft</b>, <b>Meta</b> ou encore <b>AMD</b>.
    Il couvre une période allant de janvier 2023 à aujourd’hui, avec un focus sur les indicateurs boursiers clés : prix d’ouverture, de clôture, volumes échangés, etc.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔍 Aperçu des données")
    if st.button("Afficher le dataset"):
        st.dataframe(df)

        st.markdown("### 📘 Dictionnaire des colonnes")
        st.markdown("""
        - **Date** : la date de l’enregistrement du cours boursier  
        - **Open** : prix d’ouverture du titre  
        - **High** : prix le plus haut atteint durant la journée  
        - **Low** : prix le plus bas atteint durant la journée  
        - **Close** : prix de clôture du titre  
        - **Adj Close** : prix de clôture ajusté en tenant compte des splits ou dividendes  
        - **Volume** : nombre total d’actions échangées ce jour-là  
        - **Ticker** : nom de l’entreprise (ex: NVDA, MSFT…)  
        """)


elif page == pages[2]:
    st.markdown("## 📊 Data Viz")
    st.markdown("""
    <div style='text-align: justify; font-size: 16px;'>
    Cette section présente des visualisations interactives permettant d’analyser le comportement des entreprises liées à l’intelligence artificielle sur les marchés boursiers.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("----")
    # Image d'illustration
    st.image("https://raw.githubusercontent.com/AkiraInsight/-STOCKWATCH-/main/Data%20Viz.jpg", width=800)
    
    st.markdown("### 📈 Cours de clôture moyen")
    st.markdown("""
    Le graphique ci-dessous montre l’évolution moyenne des cours de clôture pour chaque entreprise IA. 
    Cela permet de repérer les tendances et comparer les dynamiques de marché.
    """)
    
    df = pd.read_csv("https://raw.githubusercontent.com/AkiraInsight/-STOCKWATCH-/main/stockchange_ai_1y.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    avg_close = df.groupby(["Date", "Ticker"])["Close"].mean().reset_index()
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=avg_close, x="Date", y="Close", hue="Ticker", ax=ax1, linewidth=2, palette='husl')
    ax1.set_title("Évolution du cours de clôture moyen par entreprise", fontsize=14)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cours ($)")
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # un tick par mois
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # format AAAA-MM
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.grid(True)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot(fig1)

    st.markdown("""
    🔍 On remarque par exemple que certaines entreprises comme **NVIDIA** ou **Meta** présentent des hausses marquées sur certaines périodes, ce qui peut refléter un fort engouement du marché. 
    D'autres, comme **Amazon** ou **Google**, montrent des mouvements plus modérés mais constants, traduisant une stabilité plus forte.
    """)

    st.markdown("----")
    st.markdown("### 📊 Volume moyen échangé")
    st.markdown("""
    Ce graphique montre la moyenne des volumes échangés pour chaque entreprise.
    Plus le volume est élevé, plus l’action est activement tradée, ce qui peut signaler un intérêt des investisseurs.
    """)
    vol_moyen = df.groupby("Ticker")["Volume"].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=vol_moyen.index, y=vol_moyen.values, palette="Spectral", ax=ax2)
    ax2.set_title("Volume moyen échangé par entreprise", fontsize=14)
    ax2.set_xlabel("Entreprise")
    ax2.set_ylabel("Volume moyen")
    ax2.grid(axis='y')
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

    st.markdown("""
    🔍 Ce graphique met en évidence les entreprises les plus actives sur les marchés. 
    **NVIDIA** et **AMD** ont des volumes particulièrement élevés, ce qui reflète leur forte attractivité et leur rôle central dans l'écosystème IA.
    À l'inverse, des entreprises comme **IBM** ou **Intel** peuvent montrer une activité plus discrète, malgré leur rôle historique.
    """)

    st.markdown("----")
    st.markdown("### 🔄 Corrélations entre indicateurs")
    st.markdown("""
    Cette matrice de corrélation permet d’identifier les relations entre les variables clés. 
    Par exemple, une forte corrélation entre “Open” et “Close” indique que les variations journalières restent contenues.
    """)
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if 'Adj Close' in df.columns:
        numeric_cols.insert(4, 'Adj Close')
    correlation_matrix = df[numeric_cols].corr()
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="RdYlGn", fmt=".2f", linewidths=0.5, ax=ax3)
    ax3.set_title("Matrice de corrélation entre les variables numériques", fontsize=14)
    st.pyplot(fig3)

    st.markdown("""
    🔍 Cette matrice de corrélation révèle des liens forts entre les prix d'ouverture, de clôture, les prix hauts et bas du jour.
    Cela confirme une certaine cohérence dans les mouvements intra-journaliers. 
    Le volume est généralement moins corrélé avec les prix, ce qui signifie qu’il peut varier indépendamment du niveau de l’action.
    """)

    st.markdown("----")
    st.markdown("""
    <div style='text-align: justify; font-size: 16px; margin-top: 30px;'>
    Grâce à ces visualisations, on obtient un aperçu clair des dynamiques boursières des entreprises IA. 
    Ces insights permettent de détecter des opportunités ou anomalies et posent les bases d’une modélisation prédictive pertinente.
    </div>
    """, unsafe_allow_html=True)

elif page == pages[3]:
    st.markdown("## 🔧 Pré-processing des données")
    intro_image_path = "https://raw.githubusercontent.com/AkiraInsight/-STOCKWATCH-/main/Pre-Processing.jpg"
    st.image(intro_image_path, width=800)
    
    # Affichage initial du dataset
    dataset_path = "https://raw.githubusercontent.com/AkiraInsight/-STOCKWATCH-/main/stockchange_ai_1y.csv"
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        st.error(f"🚨 Erreur de chargement du dataset depuis GitHub : {e}")
        st.stop()
    
    # Étape 1: Conversion de la colonne 'Date' en datetime
    if st.button("Convertir la colonne 'Date' en datetime"):
        df['Date'] = pd.to_datetime(df['Date'])
        st.markdown("### 🕰️ Conversion de la colonne 'Date'")
        st.write("La colonne 'Date' a été convertie en format datetime pour faciliter les opérations basées sur les dates.")
        st.dataframe(df)
    
    st.markdown("---")
    
    # Étape 2: Vérification des formats numériques
    if st.button("Vérifier les formats numériques"):
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        st.markdown("### 🔢 Vérification des formats numériques")
        st.write("Les colonnes numériques comme 'Open', 'High', 'Low', 'Close', et 'Volume' ont été converties au bon format numérique.")
        st.dataframe(df)
    
    st.markdown("---")
    
    # Étape 3: Suppression des valeurs manquantes critiques
    if st.button("Supprimer les valeurs manquantes"):
        df.dropna(subset=['Date', 'Close'], inplace=True)
        st.markdown("### 🧹 Suppression des valeurs manquantes")
        st.write("Les lignes avec des valeurs manquantes dans les colonnes 'Date' et 'Close' ont été supprimées.")
        st.dataframe(df)
    
    st.markdown("---")
    
    # Étape 4: Calcul des rendements
    if st.button("Calculer les rendements quotidiens"):
        # Inspection des colonnes du DataFrame
        st.write("🔍 Colonnes du DataFrame :", df.columns.tolist())
        st.write("🧪 Aperçu de df['Close'] :", df['Close'].head())
        st.write("🧪 Type de df['Close'] :", type(df['Close']))

        # Calcul des rendements
        df['Return'] = df.groupby('Company')['Close'].transform(lambda x: x.pct_change())

        # Calcul de la volatilité sur 7 et 30 jours en utilisant la colonne 'Return'
        df['Volatility_7'] = df.groupby('Company')['Return'].transform(lambda x: x.rolling(window=7).std())
        df['Volatility_30'] = df.groupby('Company')['Return'].transform(lambda x: x.rolling(window=30).std())

        # Calcul du ratio High/Low
        df['HL_ratio'] = (df['High'] - df['Low']) / df['Low']

        # Calcul du momentum sur 7 jours
        df['Momentum_7'] = df.groupby('Company')['Close'].transform(lambda x: x - x.shift(7))

        # Supprimer les valeurs manquantes (causées par les rollings)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        st.markdown("### 📉 Calcul des rendements quotidiens")
        st.write("La colonne 'Return' a été ajoutée, indiquant le rendement quotidien de chaque action.")
        st.dataframe(df)
    
    st.markdown("---")
    
    # Étape 5: Calcul des moyennes mobiles
    if st.button("Calculer les moyennes mobiles"):
        df['MA_7'] = df.groupby('Company')['Close'].transform(lambda x: x.rolling(window=7).mean())
        df['MA_30'] = df.groupby('Company')['Close'].transform(lambda x: x.rolling(window=30).mean())
        st.markdown("### 📊 Calcul des moyennes mobiles")
        st.write("Les moyennes mobiles sur 7 et 30 jours ont été calculées pour chaque entreprise.")
        st.dataframe(df)
    
    st.markdown("---")
    
    # Étape 6: Calcul de la volatilité
    if st.button("Calculer la volatilité"):
        # Calcul des rendements si la colonne 'Return' n'existe pas encore
        if 'Return' not in df.columns:
            df['Return'] = df.groupby('Company')['Close'].pct_change()

        # Calcul de la volatilité sur 7 et 30 jours en utilisant la colonne 'Return'
        df['Volatility_7'] = df.groupby('Company')['Return'].transform(lambda x: x.rolling(window=7).std())
        df['Volatility_30'] = df.groupby('Company')['Return'].transform(lambda x: x.rolling(window=30).std())
        st.markdown("### 🌪️ Calcul de la volatilité")
        st.write("La volatilité sur 7 et 30 jours a été calculée pour chaque entreprise.")
        st.dataframe(df)
    
    st.markdown("---")
    
    # Étape 7: Calcul du ratio High/Low
    if st.button("Calculer le ratio High/Low"):
        df['HL_ratio'] = (df['High'] - df['Low']) / df['Low']
        st.markdown("### 📊 Calcul du ratio High/Low")
        st.write("Le ratio High/Low a été calculé pour chaque entreprise, indiquant la variation de prix durant la journée.")
        st.dataframe(df)
    
    st.markdown("---")
    
    # Étape 8: Calcul du momentum
    if st.button("Calculer le momentum"):
        df['Momentum_7'] = df.groupby('Company')['Close'].transform(lambda x: x - x.shift(7))
        st.markdown("### 🏃‍♂️ Calcul du momentum")
        st.write("Le momentum sur 7 jours a été calculé pour chaque action.")
        st.dataframe(df)
    
    st.markdown("---")
    
    # Étape 9: Suppression des valeurs manquantes (causées par les rollings)
    if st.button("Supprimer les valeurs manquantes (causées par les rollings)"):
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        st.markdown("### 🧹 Suppression des valeurs manquantes (causées par les rollings)")
        st.write("Les lignes avec des valeurs manquantes causées par les calculs de rolling ont été supprimées.")
        st.dataframe(df)
    
    st.markdown("---")
    

# --- Page Modélisation / Machine Learning ---
elif page == pages[4]:
    st.markdown("## 🤖 Modélisation / Machine Learning ⚙️")
    intro_image_path = "https://raw.githubusercontent.com/AkiraInsight/-STOCKWATCH-/main/Machine%20Learning.jpg"
    st.image(intro_image_path, width=800)

    if st.button("🔄 Actualiser les données et réentraîner le modèle"):
        df_csv = get_data_from_yfinance()
        df_csv['Date'] = pd.to_datetime(df_csv['Date'])
        last_date = df_csv['Date'].max()

        tickers = df_csv['Ticker'].unique()
        df_new_all = []

        for ticker in tickers:
            import datetime
            today = datetime.datetime.today()
            new_data = yf.download(ticker, start=last_date + pd.Timedelta(days=1), end=today + pd.Timedelta(days=1), interval="1d", progress=False)
            if not new_data.empty:
                new_data.reset_index(inplace=True)
                new_data["Ticker"] = ticker
                company_name = df_csv[df_csv["Ticker"] == ticker]["Company"].iloc[0]
                new_data["Company"] = company_name
                df_new_all.append(new_data)

        df_new = pd.concat(df_new_all, ignore_index=True) if df_new_all else pd.DataFrame()
        df = pd.concat([df_csv, df_new], ignore_index=True)
        # Étendre dynamiquement les données avec 7 jours futurs à chaque réentraînement
        extended_rows = []
        for company in df["Company"].unique():
            company_df = df[df["Company"] == company].copy()
            if company_df.empty:
                continue
            last_date = company_df["Date"].max()
            last_rows = company_df.sort_values("Date").copy()
            if last_rows.empty:
                continue
            for i in range(1, 8):
                new_row = last_rows.iloc[-1].copy()
                new_row["Date"] = last_date + pd.Timedelta(days=i)
                extended_rows.append(new_row)

        # Fusion et tri
        df = pd.concat([df] + [pd.DataFrame([row]) for row in extended_rows], ignore_index=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values(by=["Company", "Date"], inplace=True)
        st.markdown("### 🆕 Données récemment récupérées")
        
        # Nettoyage des colonnes si MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns]

        # Nettoyage des valeurs de Volume si chaîne avec virgules
        if "Volume" in df.columns and df["Volume"].dtype == object:
            df["Volume"] = df["Volume"].str.replace(",", "").astype(float)
        else:
            st.warning("⚠️ La colonne 'Volume' est absente ou invalide, vérifie le dataset.")
        st.write("🔍 Shape du DataFrame après mise à jour :", df.shape)
        st.write("🔍 Aperçu des premières lignes :", df.head())
        st.write("🔍 Colonnes disponibles :", df.columns.tolist())
        df.drop_duplicates(subset=['Date', 'Ticker'], inplace=True)
        df.sort_values(by=["Company", "Date"], inplace=True)

        try:
            df['Return'] = df.groupby('Company')['Close'].transform(lambda x: x.pct_change())
            df['Volatility_7'] = df.groupby('Company')['Return'].transform(lambda x: x.rolling(window=7).std())
            df['Momentum_7'] = df.groupby('Company')['Close'].transform(lambda x: x - x.shift(7))
            df.dropna(subset=['Open', 'High', 'Low', 'Volume', 'Close', 'Volatility_7', 'Momentum_7'], inplace=True)
        except Exception as e:
            st.error(f"❌ Erreur dans le feature engineering : {e}")
            st.stop()
        
        features = ['Open', 'High', 'Low', 'Volume', 'Volatility_7', 'Momentum_7']
        X = df[features]
        y = df['Close']

        # Normalisation
        scaler = StandardScaler()
        if df[features].isnull().any().any():
            st.error("❌ Certaines valeurs de features sont nulles. Vérifie le calcul de Volatility_7 ou Momentum_7.")
            st.stop()
        if X.empty:
            st.warning("⚠️ Les features sélectionnées sont vides. Vérifiez que les colonnes 'Open', 'High', 'Low', 'Volume', 'Volatility_7', 'Momentum_7' sont bien présentes et non nulles.")
            st.dataframe(df[features].head())
            st.stop()
        X_scaled = scaler.fit_transform(X)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Linear Regression Training
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 👉 Sauvegarde du modèle et du scaler
        token = "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"
        joblib.dump(df, "df.joblib")
        joblib.dump(model, "model.joblib")
        joblib.dump(scaler, "scaler.joblib")
        upload_to_github("df.joblib", "AkiraInsight/-STOCKWATCH-", "df.joblib", token)
        upload_to_github("model.joblib", "AkiraInsight/-STOCKWATCH-", "model.joblib", token)
        upload_to_github("scaler.joblib", "AkiraInsight/-STOCKWATCH-", "scaler.joblib", token)

        # ✅ Upload automatique des modèles vers GitHub si token et repo configurés

        token = "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"
        if token:
            github_repo = "AkiraInsight/-STOCKWATCH-"
            upload_to_github("model.joblib", github_repo, "model.joblib", token)
            upload_to_github("scaler.joblib", github_repo, "scaler.joblib", token)
            upload_to_github("df.joblib", github_repo, "df.joblib", token)
            st.success("✅ Modèles exportés sur GitHub.")
        else:
            st.warning("⚠️ Aucun token GitHub détecté. Vérifie que tu as bien défini GITHUB_TOKEN dans `.streamlit/secrets.toml`.")
        

        # Store in session_state
        st.session_state.update({
            'model': model,
            'scaler': scaler,
            'df': df
        })
        st.sidebar.write("🧠 Session:", list(st.session_state.keys()))

        st.success("✅ Modèle entraîné avec succès !")
        st.write(f"🔹 MSE : {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"🔹 R²  : {r2_score(y_test, y_pred):.4f}")



elif page == pages[5]:
    st.markdown("## 📈 Application – Conseils d'achat/vente")
    st.markdown("ℹ️ **Note :** Les prédictions affichées sont réalisées à **J+7**, en se basant sur les données historiques disponibles.")
    st.markdown("ℹ️ **Note :** Les prédictions affichées sont réalisées à **J+7**, en se basant sur les données historiques disponibles.")
    intro_image_path = "https://raw.githubusercontent.com/AkiraInsight/-STOCKWATCH-/main/Application.jpg"
    st.image(intro_image_path, width=800)
    
    st.markdown("""
    Cette section vous permet d’obtenir une recommandation (Acheter / Attendre / Vendre) pour chaque entreprise analysée, basée sur la prédiction du modèle.
    """, unsafe_allow_html=True)

    # Charger modèle, scaler et df depuis session_state ou fichiers locaux
    if "model" in st.session_state and "scaler" in st.session_state and "df" in st.session_state:
        model = st.session_state["model"]
        scaler = st.session_state["scaler"]
        df = st.session_state["df"]
    elif os.path.exists("model.joblib") and os.path.exists("scaler.joblib") and os.path.exists("df.joblib"):
        model = joblib.load("model.joblib")
        scaler = joblib.load("scaler.joblib")
        df = joblib.load("df.joblib")
    else:
        st.warning("🚨 Le modèle n'est pas disponible. Veuillez l'entraîner dans l'onglet 'Modélisation / Machine Learning ⚙️'.")
        st.stop()
    st.write("🔍 Date du jour :", pd.Timestamp.today().date())

    # Vérification que le DataFrame n'est pas vide
    import datetime
    today = datetime.datetime.today()
    df_updated = []
    for ticker in df["Ticker"].unique():
        company = df[df["Ticker"] == ticker]["Company"].iloc[0]
        last_date = df[df["Ticker"] == ticker]["Date"].max()
        new_data = yf.download(ticker, start=last_date + pd.Timedelta(days=1), end=today + pd.Timedelta(days=1), interval="1d", progress=False)
        if not new_data.empty:
            new_data.reset_index(inplace=True)
            new_data["Ticker"] = ticker
            new_data["Company"] = company
            df_updated.append(new_data)
        else:
            pass  # Ne rien faire si aucune nouvelle donnée, évite l'erreur d'indentation
            # st.warning(f"⚠️ Aucune donnée disponible pour {ticker} entre {last_date.date()} et {today.date()}")
    if df_updated:
        df_new = pd.concat(df_updated, ignore_index=True)
        df = pd.concat([df, df_new], ignore_index=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df.drop_duplicates(subset=['Date', 'Ticker'], inplace=True)
        df.sort_values(by=["Company", "Date"], inplace=True)
    if df.empty:
        st.error("❌ Le DataFrame est vide. Veuillez réentraîner le modèle pour générer un jeu de données valide.")
        st.stop()

    # Étendre dynamiquement les données avec 7 jours futurs à chaque lancement
    extended_rows = []
    for company in df["Company"].unique():
        company_df = df[df["Company"] == company].copy()
        if company_df.empty:
            continue
        last_row = company_df.sort_values("Date").iloc[-1].copy()
        for i in range(1, 8):
            new_row = last_row.copy()
            new_row["Date"] = last_row["Date"] + pd.Timedelta(days=i)
            extended_rows.append(new_row)

    # Fusion et tri
    df = pd.concat([df] + extended_rows, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by=["Company", "Date"], inplace=True)

    # Recalcul complet des features même pour les lignes futures
    df['Return'] = df.groupby('Company')['Close'].transform(lambda x: x.pct_change())
    df['Volatility_7'] = df.groupby('Company')['Return'].transform(lambda x: x.rolling(window=7).std())
    df['Momentum_7'] = df.groupby('Company')['Close'].transform(lambda x: x - x.shift(7))
    company_options = df["Company"].dropna().unique()
    selected_company = st.selectbox("Choisissez une entreprise", company_options)

    from datetime import timedelta, date
    
    # Dates réelles du DataFrame
    real_dates = df[df["Company"] == selected_company]["Date"].dropna().dt.date
    real_dates_list = sorted(real_dates.unique(), reverse=True)
    if real_dates_list:
        max_date = real_dates_list[0]
        extended_dates = [max_date + timedelta(days=i) for i in range(1, 8)]
    else:
        extended_dates = []
    all_dates = sorted(set(real_dates_list + extended_dates), reverse=True)
    
    available_dates = all_dates
    selected_date = st.selectbox("📆 Choisissez une date", options=[str(d) for d in available_dates])
    
    df["Date"] = pd.to_datetime(df["Date"])
    filtered_df = df[(df["Company"] == selected_company) & (df["Date"].dt.date.astype(str) == selected_date)]

    required_cols = ['Open', 'High', 'Low', 'Volume', 'Company', 'Ticker']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Colonnes manquantes pour la prédiction : {missing_cols}")
        st.stop()
    features = ['Open', 'High', 'Low', 'Volume']
    cat_features = ['Company', 'Ticker']

    if not filtered_df.empty:
        # Calcul de Volatility_7 et Momentum_7 pour la prédiction si manquants
        if 'Return' not in filtered_df.columns:
            filtered_df['Return'] = filtered_df.groupby('Company')['Close'].transform(lambda x: x.pct_change())
        if 'Volatility_7' not in filtered_df.columns:
            filtered_df['Volatility_7'] = filtered_df.groupby('Company')['Return'].transform(lambda x: x.rolling(window=7).std())
        if 'Momentum_7' not in filtered_df.columns:
            filtered_df['Momentum_7'] = filtered_df.groupby('Company')['Close'].transform(lambda x: x - x.shift(7))
 
        filtered_df.dropna(subset=['Volatility_7', 'Momentum_7'], inplace=True)
 
        features = ['Open', 'High', 'Low', 'Volume', 'Volatility_7', 'Momentum_7']
        cat_features = ['Company', 'Ticker']
 
        X_app = filtered_df[features + cat_features].copy()
        y_real = filtered_df["Close"]
    else:
        st.warning("🚨 Aucune donnée disponible pour cette entreprise à cette date.")
        st.stop()

    # One-hot encoding
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    X_cat = ohe.fit_transform(X_app[cat_features])
    ohe_df = pd.DataFrame(X_cat, columns=ohe.get_feature_names_out(cat_features), index=X_app.index)

    X_app_final = pd.concat([X_app[features], ohe_df], axis=1)
    X_app_scaled = scaler.transform(X_app_final)
    y_pred = model.predict(X_app_scaled)

    reco = []
    for real, pred in zip(y_real, y_pred):
        if pred > real * 1.02:
            reco.append("🟢 Acheter")
        elif pred < real * 0.98:
            reco.append("🔴 Vendre")
        else:
            reco.append("🟡 Attendre")

    last_data = filtered_df.copy()
    last_data["Prix Réel"] = y_real
    last_data["Prix Prédit"] = y_pred
    last_data["Conseil"] = reco

    st.dataframe(last_data[["Company", "Ticker", "Prix Réel", "Prix Prédit", "Conseil"]])
