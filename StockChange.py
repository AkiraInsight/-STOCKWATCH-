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

# --- STYLE & SIDEBAR ---

st.sidebar.image("Logo Akira Insight.png", width=150)
# Sidebar Title & Info
st.sidebar.title("📊 StockChange")
st.sidebar.write("📆 Promotion DataScientest - MLOps 2025")

# Navigation
pages = [
    "Introduction",
    "Présentation du Dataset",
    "Data Viz 📊",
    "Pré-processing 👷‍♂️",
    "Modélisation / Machine Learning ⚙️",
    "Application 🎬"
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
    </style>
    """,
    unsafe_allow_html=True
)

if page == pages[0]:
    st.markdown("## 🎯 Introduction – StockChange")
    intro_image_path = "Introduction.jpg"
    if os.path.exists(intro_image_path):
        st.image(intro_image_path, width=800)
    else:
        st.warning("🚨 L'image d'intro n'a pas été trouvée.")
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

    dataset_path = "stockchange_ai_1y.csv"
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        
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
    else:
        st.warning("🚨 Le fichier de données n’a pas été trouvé. Veuillez vérifier le chemin.")

elif page == pages[2]:
    st.markdown("## 📊 Data Viz")
    st.markdown("""
    <div style='text-align: justify; font-size: 16px;'>
    Cette section présente des visualisations interactives permettant d’analyser le comportement des entreprises liées à l’intelligence artificielle sur les marchés boursiers.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("----")
    # Image d'illustration
    if os.path.exists("Data Viz.jpg"):
        st.image("Data Viz.jpg", use_container_width=True, width=800)
        
        st.markdown("### 📈 Cours de clôture moyen")
        st.markdown("""
        Le graphique ci-dessous montre l’évolution moyenne des cours de clôture pour chaque entreprise IA. 
        Cela permet de repérer les tendances et comparer les dynamiques de marché.
        """)
    
    df = pd.read_csv("stockchange_ai_1y.csv")
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
    intro_image_path = "Pre-Processing.jpg"
    if os.path.exists(intro_image_path):
        st.image(intro_image_path, width=800)
    else:
        st.warning("🚨 L'image Pre-Processing.jpg n'a pas été trouvée.")
    
    # Affichage initial du dataset
    dataset_path = "stockchange_ai_1y.csv"
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
    else:
        st.warning("🚨 Le fichier de données n’a pas été trouvé. Veuillez vérifier le chemin.")
    
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
        # Calcul des rendements
        df['Return'] = df.groupby('Company')['Close'].pct_change()

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
elif page == pages[4]:  # Modélisation / Machine Learning
    st.markdown("## 🤖 Modélisation / Machine Learning ⚙️")
    intro_image_path = "Machine Learning.jpg"
    if os.path.exists(intro_image_path):
        st.image(intro_image_path, width=800)

    dataset_path = "stockchange_ai_1y.csv"
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)

        if st.button("Lancer le pipeline complet"):
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date', 'Close'], inplace=True)

            features = ['Open', 'High', 'Low', 'Volume']
            target = 'Close'
            cat_features = ['Company', 'Ticker']

            X = df[features + cat_features].copy()
            y = df[target]

            preprocessor = ColumnTransformer(
                transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)],
                remainder='passthrough'
            )

            pipeline = Pipeline(steps=[
                ('preprocessing', preprocessor),
                ('model', LinearRegression())
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            st.write("📊 Résultats du modèle :")
            st.write(f"🔹 MSE : {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"🔹 R²  : {r2_score(y_test, y_pred):.4f}")

            # Graphique 1 - Prédiction
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(y_test.values, label="Réel", color='blue')
            ax1.plot(y_pred, label="Prédit", color='orange')
            ax1.set_title("📈 Prédiction du prix de clôture - Régression Linéaire")
            ax1.set_xlabel("Index")
            ax1.set_ylabel("Prix de clôture")
            ax1.legend()
            ax1.grid(True)
            st.pyplot(fig1)

            # Graphique 2 - Erreur de prédiction
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(y_test.values - y_pred, color='red', label="Erreur (réel - prédit)")
            ax2.set_title("📉 Erreurs de prédiction")
            ax2.axhline(0, linestyle='--', color='gray')
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)
            
            # Graphique 3 - Distribution des erreurs
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            sns.histplot(y_test.values - y_pred, bins=50, kde=True, color="purple", ax=ax3)
            ax3.set_title("Distribution des erreurs de prédiction")
            ax3.set_xlabel("Erreur (Réel - Prédit)")
            ax3.set_ylabel("Fréquence")
            ax3.grid(True)
            st.pyplot(fig3)
            
            # Graphique 4 - Réel vs Prédit (scatter plot)
            fig4, ax4 = plt.subplots(figsize=(6, 6))
            ax4.scatter(y_test, y_pred, alpha=0.5, color="green")
            ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
            ax4.set_xlabel("Prix réel")
            ax4.set_ylabel("Prix prédit")
            ax4.set_title("Comparaison Réel vs Prédit")
            ax4.grid(True)
            st.pyplot(fig4)

            joblib.dump(pipeline, "linear_regression_model.joblib")
            st.success("💾 Modèle sauvegardé sous 'linear_regression_model.joblib'")

    else:
        st.warning("🚨 Le fichier de données n’a pas été trouvé.")

elif page == pages[5]:
    st.markdown("## 🎬 Application – Conseils d'achat/vente")
    
    st.markdown("""
    Cette section vous permet d’obtenir une recommandation (Acheter / Attendre / Vendre) pour chaque entreprise analysée, basée sur la prédiction du modèle.
    """, unsafe_allow_html=True)
    
    # Charger le modèle
    model_path = "linear_regression_model.joblib"
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
    
        # Charger les données
        dataset_path = "stockchange_ai_1y.csv"
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.dropna(subset=['Date', 'Close'], inplace=True)
            
            # Étendre les dates jusqu'à +7 jours pour chaque entreprise
            last_dates = df.groupby("Company")["Date"].max().reset_index()
            
            for _, row in last_dates.iterrows():
                company = row["Company"]
                ticker = df[df["Company"] == company]["Ticker"].iloc[0]
                base_row = df[(df["Company"] == company) & (df["Date"] == row["Date"])].iloc[0]
                for i in range(1, 8):
                    future_row = base_row.copy()
                    future_row["Date"] = base_row["Date"] + pd.Timedelta(days=i)
                    df = pd.concat([df, pd.DataFrame([future_row])], ignore_index=True)
            
            company_options = df["Company"].dropna().unique()
            selected_company = st.selectbox("Choisissez une entreprise", company_options)
            
            available_dates = sorted(df[df["Company"] == selected_company]["Date"].dropna().dt.date.unique())
            selected_date = st.date_input("Choisissez une date", min_value=min(available_dates), max_value=max(available_dates))
            
            # Filtrer les données selon l'entreprise et la date
            filtered_df = df[(df["Company"] == selected_company) & (df["Date"] == pd.to_datetime(selected_date))]
    
            features = ['Open', 'High', 'Low', 'Volume']
            cat_features = ['Company', 'Ticker']
            if not filtered_df.empty:
                X_app = filtered_df[features + cat_features].copy()
                y_real = filtered_df["Close"]
            else:
                st.warning("🚨 Aucune donnée disponible pour cette entreprise à cette date.")
                st.stop()
    
            y_pred = pipeline.predict(X_app)
    
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
        else:
            st.warning("🚨 Fichier de données non trouvé.")
    else:
        st.warning("🚨 Modèle non trouvé. Veuillez lancer la modélisation avant.")
