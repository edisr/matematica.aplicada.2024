import pandas as pd
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from nltk.data import find
import time
from skfuzzy import membership
import numpy as np
from scipy.integrate import simps

# Verifica si el léxico ya está descargado
try:
    find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Inicializa analizador de sentimientos
sia = SentimentIntensityAnalyzer()

# Función de preprocesamiento de cada tweet
def preprocesamiento(tweet):
    tweet = re.sub(r'http\S+|www\S+|https\S+|@\w+', '', tweet, flags=re.MULTILINE) 
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()

    palabrasCont = {
        "he's": "he is", "she's": "she is", "they're": "they are", "we're": "we are", "you're": "you are",
        "I've": "I have", "you've": "you have", "we've": "we have", "they've": "they have",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
        "shouldn't": "should not", "couldn't": "could not", "mightn't": "might not", "mustn't": "must not",
        "can't": "cannot", "won't": "will not", "don't": "do not", "I'm": "I am", "it's": "it is"
    }
    for contraccion, full_form in palabrasCont.items():
        tweet = tweet.replace(contraccion, full_form)
    
    return tweet

# Cargar datos
df = pd.read_csv('test_data.csv')
df['preprocesamiento'] = df['sentence'].apply(preprocesamiento)

# Definir tiempos y variables de conteo
start_time_total = time.time()
tiempos_sentimientos = []
nropositivo, nronegativo, nroneutro = 0, 0, 0

# Función de fuzzificación triangular
def triangular_membership(x, d, e, f):
    if x < d:
        return 0
    elif d <= x < e:
        return (x - d) / (e - d)
    elif e <= x < f:
        return (f - x) / (f - e)
    else:
        return 0

# Definir parámetros de fuzzificación
d_low, e_low, f_low = 0.0, 0.2, 0.5
d_med, e_med, f_med = 0.3, 0.5, 0.8
d_high, e_high, f_high = 0.6, 0.8, 1.0

# Rango de salida y funciones de membresía
x_op = np.arange(0, 11, 1)
op_neg = membership.trimf(x_op, [0, 0, 5])
op_neu = membership.trimf(x_op, [0, 5, 10])
op_pos = membership.trimf(x_op, [5, 10, 10])

def evaluate_rules(row): 
    bajo_pos = triangular_membership(row['puntaje positivo'], d_low, e_low, f_low)
    med_pos = triangular_membership(row['puntaje positivo'], d_med, e_med, f_med)
    alto_pos = triangular_membership(row['puntaje positivo'], d_high, e_high, f_high)
    
    bajo_neg = triangular_membership(row['puntaje negativo'], d_low, e_low, f_low)
    med_neg = triangular_membership(row['puntaje negativo'], d_med, e_med, f_med)
    alto_neg = triangular_membership(row['puntaje negativo'], d_high, e_high, f_high)
    
    rules = [np.fmin(bajo_pos, bajo_neg), 
             np.fmin(med_pos, bajo_neg), 
             np.fmin(alto_pos, bajo_neg),
             np.fmin(bajo_pos, med_neg), 
             np.fmin(med_pos, med_neg), 
             np.fmin(alto_pos, med_neg),
             np.fmin(bajo_pos, alto_neg),
             np.fmin(med_pos, alto_neg), 
             np.fmin(alto_pos, alto_neg)]
    
    activation_neg = np.fmax(np.fmax(rules[3], rules[6]), rules[7]) 
    activation_neu = np.fmax(np.fmax(rules[0], rules[4]), rules[8]) 
    activation_pos = np.fmax(np.fmax(rules[1], rules[2]), rules[5]) 
    
    op_activation_low = np.fmin(activation_neg, op_neg) 
    op_activation_med = np.fmin(activation_neu, op_neu) 
    op_activation_high = np.fmin(activation_pos, op_pos) 

    aggregated = np.fmax(np.fmax(op_activation_low, op_activation_med), op_activation_high)
    COA = centroid_defuzzification(x_op, aggregated)
    
    if 0 <= COA < 3.3:
        return "Negativo"
    elif 3.3 <= COA < 6.7:
        return "Neutro"
    else:
        return "Positivo"

def centroid_defuzzification(x, mu):
    numerator = simps(x * mu, x)
    denominator = simps(mu, x)
    return numerator / denominator if denominator != 0 else 0

# Procesamiento de cada tweet
resultados = []
for idx, row in df.iterrows():
    start_time = time.time()
    puntaje = sia.polarity_scores(row['preprocesamiento'])
    row['puntaje positivo'] = puntaje['pos']
    row['puntaje negativo'] = puntaje['neg']
    sentiment = evaluate_rules(row)
    tiempo_ejecucion = time.time() - start_time

    if sentiment == 'Positivo':
        nropositivo += 1
    elif sentiment == 'Negativo':
        nronegativo += 1
    else:
        nroneutro += 1

    resultados.append([
        row['sentence'], row.get('label', ''), puntaje['pos'], puntaje['neg'], sentiment, tiempo_ejecucion
    ])
    tiempos_sentimientos.append(tiempo_ejecucion)

# Guardar resultados
df_resultados = pd.DataFrame(resultados, columns=[
    'Oración original', 'label original','Puntaje Positivo', 'Puntaje Negativo', 'Resultado de inferencia', 'tiempo de ejecución'
])
df_resultados.to_csv('resultados_con_defuzzificacion.csv', index=False)

# Reporte final
tiempo_total = time.time() - start_time_total
tiempo_promedio = np.mean(tiempos_sentimientos)

print("Sentimientos Positivos:", nropositivo)
print("Sentimientos Negativos:", nronegativo)
print("Sentimientos Neutros:", nroneutro)
print(f"Tiempo total de ejecución: {tiempo_total:.4f} segundos")
print(f"Tiempo promedio de inferencia por tweet: {tiempo_promedio:.4f} segundos")
