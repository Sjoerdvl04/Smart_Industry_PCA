import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from time import time
import warnings
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from pyod.models.pca import PCA

warnings.filterwarnings('ignore')
register_matplotlib_converters()

blog_post = st.sidebar.radio('Blogcategoriëen',
                             ['Introductie', 'Baseline methode', 'PCA methode'])

def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')

df = pd.read_csv('freezerdata_clean 15-03 tm 19-03.csv', index_col=0, parse_dates=[1])
df = df.squeeze()
outliers = pd.read_csv('freezerdata_clean_outliers.csv')

# ======================================================================================================================================================================

if blog_post == 'Introductie':
    st.header('Introductie', divider='grey')
    st.markdown("""
                Auteurs: Matthijs van Balen, Tim Lind, Sem Vredevoort en Sjoerd van Leeuwen (Tata Steel)
                """)
    
    st.markdown("""                
In dit dashboard wordt er gekeken naar data van een vriezer en of het mogelijk is om de koeltemperatuur te verschillen. Daarnaast zal er ook worden gekeken naar uitschieters. Vervolgens wordt middels matrixen geëvalueerd hoe goed de modellen (Baseline en PCA) zijn.
                """)

# ======================================================================================================================================================================

elif blog_post == 'Baseline methode':
    st.header('Baseline methode', divider='grey')
    st.markdown("""
Allereerst wordt er naar een gedeelte van de dataset gekeken. Dit is een sample van 300 minuten. Vervolgens worden er twee boxplots gemaakt van de pieken en dalen in deze sample.
                """)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    start_date = datetime(2023,3,16,23,0)
    end_date = datetime(2023,3,17,4,0)
    lim_df = df[start_date:end_date]

    time_elapsed_minutes = (lim_df.index - lim_df.index[0]).total_seconds() / 60

    peaks,_ = find_peaks(lim_df['Refrigerated'], prominence=5)
    valleys,_ = find_peaks(-lim_df['Refrigerated'],prominence=5)

    peak_rows = lim_df.iloc[peaks]
    valley_rows = lim_df.iloc[valleys]
    peak_valley = pd.concat([peak_rows, valley_rows]).sort_index()
    peak_valley['top of dal'] = np.where(peak_valley['Refrigerated'] < 0, 'dal', 'top')

    plt.subplot(1, 2, 1)
    plt.boxplot(peak_rows['Refrigerated'], vert=True)
    plt.ylabel('Temperature Refrigerated')
    plt.title('Boxplot van de pieken')

    plt.subplot(1, 2, 2)
    plt.boxplot(valley_rows['Refrigerated'], vert=True)
    plt.title('Boxplot van de dalen')

    plt.tight_layout()
    st.pyplot(plt)

    st.subheader('Pieken en dalen', divider='grey')
    st.markdown("""
Deze pieken en dalen vallen binnen een bepaalde afstand. Hiervoor is de interkwartielafstand gebruikt. In het figuur hieronder is te zien dat er geen uitschieters in pieken zitten, maar wel in de dalen.
                """)
    
    Q1 = lim_df['Refrigerated'].quantile(0.25)
    Q3 = lim_df['Refrigerated'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier = lim_df[(lim_df['Refrigerated'] < lower_bound) | (lim_df['Refrigerated'] > upper_bound)]


    plt.figure(figsize=(12, 6))
    plt.plot(lim_df.index, lim_df['Refrigerated'], label='Refrigerated Data')
    plt.plot(outlier.index, outlier['Refrigerated'], "ro", label='Outliers')
    plt.axhline(y=lower_bound, color='g', linestyle='--', label='Lower Bound')
    plt.axhline(y=upper_bound, color='b', linestyle='--', label='Upper Bound')
    plt.xlabel("Index")
    plt.ylabel("Refrigerated Values")
    plt.title("Refrigerated Data with Outliers Highlighted and Boundaries")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.subheader('Voorspelling pieken en dalen', divider='grey')
    st.markdown("""
                """)
    X = lim_df.drop('Refrigerated', axis=1)  
    y = lim_df['Refrigerated']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    datasets = ['Trainset', 'Testset']
    datasets_selected = st.selectbox('Selecteer een set', datasets)

    if datasets_selected == 'Trainset':
        st.markdown("""
Om een eerste indruk te krijgen van de train dataset, zijn de eerste vijf regels van de dataset hieronder te zien. Aan de hand van deze variabelen gaat de koeltemperatuur voorspeld proberen te worden in het trainmodel.
                    """)
        st.table(X_train.head())
    
    elif datasets_selected == 'Testset':
        st.markdown("""
Om een eerste indruk te krijgen van de test dataset, zijn de eerste vijf regels van de dataset hieronder te zien. Aan de hand van deze variabelen gaat de koeltemperatuur voorspeld proberen te worden in het testmodel.
                    """)
        st.table(X_test.head())

    st.subheader('Evaluatie voorspelling', divider='grey')
    st.markdown("""
In de confusion matrix hieronder is te zien dat dit baseline model goed heeft gewerkt. Zo identificeerde het 58 niet-uitschieters correct 2 uitschieters correct.
                """)
    Q1_train = y_train.quantile(0.25)
    Q3_train = y_train.quantile(0.75)
    IQR_train = Q3_train - Q1_train
    lower_bound_train = Q1_train - 1.5 
    upper_bound_train = Q3_train + 1.5 

    true_outliers = (y_test < lower_bound_train) | (y_test > upper_bound_train)

    Q1_test = y_test.quantile(0.25)
    Q3_test = y_test.quantile(0.75)
    IQR_test = Q3_test - Q1_test
    lower_bound_test = Q1_test - 1.5 
    upper_bound_test = Q3_test + 1.5 

    predicted_outliers = (y_test < lower_bound_test) | (y_test > upper_bound_test)

    cm = confusion_matrix(true_outliers, predicted_outliers)
    tn, fp, fn, tp = cm.ravel()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Outlier', 'Outlier'])
    disp.plot(cmap='Blues')
    st.pyplot(plt)
    st.write(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")


# ======================================================================================================================================================================

elif blog_post == 'PCA methode':
    st.header('PCA methode', divider='grey')
    st.markdown("""
Een andere methode om uitschieters te identificeren is de PCA (Principal Component Analysis). Hiervoor is ook weer gebruik gemaakt van een train, test split op de data. Allereest zijn alle NaN-waarden geinterpoleerd en daarna is de dataset opgesplitst in een 80-20 format.

Ook is er gebruik gemaakt van verschillende contamination rates. Dit is het percentage verwachte outliers in de dataset. Wij hebben 0.01, 0.05, 0.1, 0.15 gebruikt als mogelijke contamination rates. 
                """)
    df = pd.read_csv('freezerdata_clean_outliers.csv')
    df2 = df.select_dtypes(include=[np.number]).diff()
    df2 = df2.fillna(method='ffill') 
    df2 = df2.fillna(method='bfill')

    features = ["Door", "Environment", "HotGasPipe", "LiquidPipe", "Refrigerated", "SuctionPipe"]

    X = df2[features].values
    y = df2['Refrigerated'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    contamination_rates = [0.01, 0.05, 0.1, 0.15]
    contamination_rates_selected = st.selectbox('Selecteer een contamination rate', contamination_rates)

    def plot_results(X, predictions, title):
        pca_viz = PCA(contamination=0.05)
        pca_viz.fit(X)
        X_transformed = pca_viz.detector_.transform(X)
        
        plt.figure(figsize=(10, 6))
        
        plt.scatter(X_transformed[predictions == 0, 0],
                X_transformed[predictions == 0, 1],
                c='blue', label='Normaal', alpha=0.5)
        plt.scatter(X_transformed[predictions == 1, 0],
                X_transformed[predictions == 1, 1],
                c='red', label='Outlier', alpha=0.7)
        
        plt.title(title)
        plt.xlabel('Eerste hoofdcomponent')
        plt.ylabel('Tweede hoofdcomponent')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)    

    if contamination_rates_selected == 0.01:
        pca = PCA(contamination=contamination_rates_selected)
        pca.fit(X_train)

        y_test_binary = (y_test > 0.5).astype(int)
        
        y_pred = pca.predict(X_test)
        
        print(f"\nContamination rate: {contamination_rates_selected}")
        print(f"Aantal outliers: {sum(y_pred)}")
        print(f"Percentage outliers: {(sum(y_pred)/len(y_pred))*100:.1f}%")

        cm = confusion_matrix(y_test_binary, y_pred, labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normaal", "Outlier"])

        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix (Contamination rate: {contamination_rates_selected})")
        st.pyplot(plt)

        plot = plot_results(X_test, y_pred, f'Outlier Detectie (contamination rate {contamination_rates_selected})')

    elif contamination_rates_selected == 0.05:
        pca = PCA(contamination=contamination_rates_selected)
        pca.fit(X_train)

        y_test_binary = (y_test > 0.5).astype(int)
        
        y_pred = pca.predict(X_test)
        
        print(f"\nContamination rate: {contamination_rates_selected}")
        print(f"Aantal outliers: {sum(y_pred)}")
        print(f"Percentage outliers: {(sum(y_pred)/len(y_pred))*100:.1f}%")

        cm = confusion_matrix(y_test_binary, y_pred, labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normaal", "Outlier"])

        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix (Contamination rate: {contamination_rates_selected})")
        st.pyplot(plt)

        plot = plot_results(X_test, y_pred, f'Outlier Detectie (contamination rate {contamination_rates_selected})')

    elif contamination_rates_selected == 0.1:
        pca = PCA(contamination=contamination_rates_selected)
        pca.fit(X_train)

        y_test_binary = (y_test > 0.5).astype(int)
        
        y_pred = pca.predict(X_test)
        
        print(f"\nContamination rate: {contamination_rates_selected}")
        print(f"Aantal outliers: {sum(y_pred)}")
        print(f"Percentage outliers: {(sum(y_pred)/len(y_pred))*100:.1f}%")

        cm = confusion_matrix(y_test_binary, y_pred, labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normaal", "Outlier"])

        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix (Contamination rate: {contamination_rates_selected})")
        st.pyplot(plt)

        plot = plot_results(X_test, y_pred, f'Outlier Detectie (contamination rate {contamination_rates_selected})')

    elif contamination_rates_selected == 0.15:
        pca = PCA(contamination=contamination_rates_selected)
        pca.fit(X_train)

        y_test_binary = (y_test > 0.5).astype(int)
        
        y_pred = pca.predict(X_test)
        
        print(f"\nContamination rate: {contamination_rates_selected}")
        print(f"Aantal outliers: {sum(y_pred)}")
        print(f"Percentage outliers: {(sum(y_pred)/len(y_pred))*100:.1f}%")

        cm = confusion_matrix(y_test_binary, y_pred, labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normaal", "Outlier"])

        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix (Contamination rate: {contamination_rates_selected})")
        st.pyplot(plt)

        plot = plot_results(X_test, y_pred, f'Outlier Detectie (contamination rate {contamination_rates_selected})')



