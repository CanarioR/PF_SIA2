"""
================================================================================
PROYECTO: DIAGNÓSTICO DE SALUD FINANCIERA
Sistema de IA para evaluar la aprobación de créditos bancarios
usando Árboles de Decisión
================================================================================
"""

# Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("SISTEMA DE DIAGNÓSTICO DE SALUD FINANCIERA")
print("Modelo de IA para Aprobación de Créditos Bancarios")
print("=" * 80)
print()

# Generación de dataset sintético realista
print("PASO 1: Generando dataset sintético...")
print("-" * 80)

np.random.seed(42)
n_samples = 500

def generar_dataset(n):
    """
    Genera un dataset sintético con reglas de negocio realistas
    para la aprobación de créditos bancarios.
    """
    data = []
    
    for i in range(n):
        edad = np.random.randint(18, 70)
        
        # Ingresos correlacionados con edad
        if edad < 25:
            ingresos = np.random.normal(800, 300)
        elif edad < 35:
            ingresos = np.random.normal(1500, 500)
        elif edad < 50:
            ingresos = np.random.normal(2500, 800)
        else:
            ingresos = np.random.normal(2000, 600)
        
        ingresos = max(300, ingresos)
        
        # Score crediticio (300-850, estándar FICO)
        score_base = np.random.normal(650, 100)
        score_crediticio = int(np.clip(score_base, 300, 850))
        
        # Deuda total relacionada con ingresos y score
        if score_crediticio > 700:
            deuda = np.random.uniform(0, ingresos * 2)
        elif score_crediticio > 600:
            deuda = np.random.uniform(0, ingresos * 3)
        else:
            deuda = np.random.uniform(ingresos * 0.5, ingresos * 5)
        
        antiguedad_credito = min(edad - 18, np.random.randint(0, 30))
        num_cuentas = np.random.randint(1, 6)
        ratio_deuda_ingreso = deuda / ingresos if ingresos > 0 else 5
        
        # Lógica de aprobación
        aprobado = 0
        
        if score_crediticio >= 700 and ratio_deuda_ingreso < 2:
            aprobado = 1
        elif score_crediticio >= 650 and ratio_deuda_ingreso < 1.5 and ingresos > 1500:
            aprobado = 1
        elif score_crediticio >= 600 and ratio_deuda_ingreso < 1 and antiguedad_credito > 5:
            aprobado = 1
        
        if deuda > ingresos * 4 or score_crediticio < 500:
            aprobado = 0
        
        # Ruido realista (10% de casos atípicos)
        if np.random.random() < 0.1:
            aprobado = 1 - aprobado
        
        data.append({
            'Edad': edad,
            'Ingresos_Mensuales': round(ingresos, 2),
            'Deuda_Total': round(deuda, 2),
            'Score_Crediticio': score_crediticio,
            'Antiguedad_Credito_Anos': antiguedad_credito,
            'Num_Cuentas': num_cuentas,
            'Ratio_Deuda_Ingreso': round(ratio_deuda_ingreso, 2),
            'Aprobado': aprobado
        })
    
    return pd.DataFrame(data)

df = generar_dataset(n_samples)

print(f"Dataset generado exitosamente: {len(df)} registros")
print()
print("Primeras 10 filas del dataset:")
print(df.head(10))
print()
print("Información del dataset:")
print(df.info())
print()
print("Estadísticas descriptivas:")
print(df.describe())
print()

print("Distribución de créditos:")
print(df['Aprobado'].value_counts())
print(f"Aprobados: {df['Aprobado'].sum()} ({df['Aprobado'].mean()*100:.1f}%)")
print(f"Rechazados: {(1-df['Aprobado']).sum()} ({(1-df['Aprobado']).mean()*100:.1f}%)")
print()

# Preparación de datos
print("=" * 80)
print("PASO 2: Preparando datos para entrenamiento...")
print("-" * 80)

X = df.drop('Aprobado', axis=1)
y = df['Aprobado']

print(f"Características (Features): {list(X.columns)}")
print(f"Variable objetivo (Target): Aprobado (0=No, 1=Sí)")
print()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Datos divididos:")
print(f"   - Entrenamiento: {len(X_train)} registros ({len(X_train)/len(df)*100:.1f}%)")
print(f"   - Prueba: {len(X_test)} registros ({len(X_test)/len(df)*100:.1f}%)")
print()

# Entrenamiento del modelo
print("=" * 80)
print("PASO 3: Entrenando el modelo de Árbol de Decisión...")
print("-" * 80)

modelo = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

modelo.fit(X_train, y_train)

print("Modelo entrenado exitosamente!")
print(f"   - Profundidad del árbol: {modelo.get_depth()}")
print(f"   - Número de hojas: {modelo.get_n_leaves()}")
print()

# Evaluación del modelo
print("=" * 80)
print("PASO 4: Evaluando el rendimiento del modelo...")
print("-" * 80)

y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

print("MÉTRICAS DE RENDIMIENTO:")
print(f"   - Precisión en Entrenamiento: {accuracy_train*100:.2f}%")
print(f"   - Precisión en Prueba: {accuracy_test*100:.2f}%")
print(f"   - Precisión (Precision): {precision*100:.2f}%")
print(f"   - Sensibilidad (Recall): {recall*100:.2f}%")
print(f"   - F1-Score: {f1*100:.2f}%")
print()

print("REPORTE DE CLASIFICACIÓN DETALLADO:")
print("-" * 80)
print(classification_report(y_test, y_pred_test, 
                          target_names=['Rechazado (0)', 'Aprobado (1)']))
print()

# Visualización de la matriz de confusión
print("=" * 80)
print("PASO 5: Generando visualizaciones...")
print("-" * 80)

fig = plt.figure(figsize=(18, 12))

# Matriz de Confusión
ax1 = plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Rechazado', 'Aprobado'],
            yticklabels=['Rechazado', 'Aprobado'])
plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')
plt.ylabel('Valor Real')
plt.xlabel('Predicción del Modelo')

verdaderos_negativos = cm[0,0]
falsos_positivos = cm[0,1]
falsos_negativos = cm[1,0]
verdaderos_positivos = cm[1,1]

texto_explicacion = f"""
VN: {verdaderos_negativos} (Rechazos correctos)
FP: {falsos_positivos} (Aprobó erróneamente)
FN: {falsos_negativos} (Rechazó erróneamente)
VP: {verdaderos_positivos} (Aprobaciones correctas)
"""
plt.text(1.5, -0.5, texto_explicacion, fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Importancia de las Variables
ax2 = plt.subplot(2, 3, 2)
importancias = modelo.feature_importances_
indices = np.argsort(importancias)[::-1]
nombres_features = X.columns

colores = plt.cm.viridis(np.linspace(0, 1, len(indices)))
bars = plt.barh(range(len(indices)), importancias[indices], color=colores)
plt.yticks(range(len(indices)), [nombres_features[i] for i in indices])
plt.xlabel('Importancia', fontweight='bold')
plt.title('Importancia de Variables en la Decisión', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontsize=9)

# Distribución de Score Crediticio por Decisión
ax3 = plt.subplot(2, 3, 3)
df_aprobados = df[df['Aprobado'] == 1]['Score_Crediticio']
df_rechazados = df[df['Aprobado'] == 0]['Score_Crediticio']

plt.hist([df_rechazados, df_aprobados], bins=20, label=['Rechazados', 'Aprobados'],
         color=['#ff6b6b', '#4ecdc4'], alpha=0.7, edgecolor='black')
plt.xlabel('Score Crediticio', fontweight='bold')
plt.ylabel('Frecuencia', fontweight='bold')
plt.title('Distribución de Score Crediticio', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Ratio Deuda/Ingreso vs Aprobación
ax4 = plt.subplot(2, 3, 4)
scatter1 = plt.scatter(df[df['Aprobado']==0]['Ratio_Deuda_Ingreso'], 
                      df[df['Aprobado']==0]['Score_Crediticio'],
                      alpha=0.6, c='#ff6b6b', label='Rechazado', s=50, edgecolors='black', linewidth=0.5)
scatter2 = plt.scatter(df[df['Aprobado']==1]['Ratio_Deuda_Ingreso'], 
                      df[df['Aprobado']==1]['Score_Crediticio'],
                      alpha=0.6, c='#4ecdc4', label='Aprobado', s=50, edgecolors='black', linewidth=0.5)
plt.xlabel('Ratio Deuda/Ingreso', fontweight='bold')
plt.ylabel('Score Crediticio', fontweight='bold')
plt.title('Relación entre Variables Clave', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Distribución de Ingresos por Decisión
ax5 = plt.subplot(2, 3, 5)
df.boxplot(column='Ingresos_Mensuales', by='Aprobado', ax=ax5,
           patch_artist=True, 
           boxprops=dict(facecolor='lightblue', color='black'),
           medianprops=dict(color='red', linewidth=2))
plt.xlabel('Decisión (0=Rechazado, 1=Aprobado)', fontweight='bold')
plt.ylabel('Ingresos Mensuales ($)', fontweight='bold')
plt.title('Ingresos Mensuales por Decisión', fontsize=14, fontweight='bold')
plt.suptitle('')

# Métricas de Rendimiento
ax6 = plt.subplot(2, 3, 6)
metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
valores = [accuracy_test, precision, recall, f1]
colores_metricas = ['#1abc9c', '#3498db', '#9b59b6', '#e74c3c']

bars = plt.bar(metricas, valores, color=colores_metricas, edgecolor='black', linewidth=2)
plt.ylim(0, 1.1)
plt.ylabel('Score', fontweight='bold')
plt.title('Métricas de Evaluación del Modelo', fontsize=14, fontweight='bold')
plt.axhline(y=0.8, color='red', linestyle='--', label='Umbral mínimo (80%)')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

for i, (bar, valor) in enumerate(zip(bars, valores)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{valor:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('c:\\Users\\diego.sanchez\\Documents\\l\\evaluacion_modelo.png', dpi=300, bbox_inches='tight')
print("Gráficos guardados en: evaluacion_modelo.png")
plt.show()

# Visualización del Árbol de Decisión
print()
print("=" * 80)
print("PASO 6: Visualizando el Árbol de Decisión...")
print("-" * 80)

fig, ax = plt.subplots(figsize=(25, 15))
plot_tree(modelo, 
          feature_names=X.columns,
          class_names=['Rechazado', 'Aprobado'],
          filled=True,
          rounded=True,
          fontsize=10,
          ax=ax)
plt.title('Árbol de Decisión Completo - Sistema de Aprobación de Créditos', 
          fontsize=18, fontweight='bold', pad=20)
plt.savefig('c:\\Users\\diego.sanchez\\Documents\\l\\arbol_decision.png', dpi=300, bbox_inches='tight')
print("Árbol de decisión guardado en: arbol_decision.png")
plt.show()

# Reglas de decisión en formato texto
print()
print("=" * 80)
print("PASO 7: Extrayendo reglas de decisión...")
print("-" * 80)

reglas = export_text(modelo, feature_names=list(X.columns))
print("REGLAS DEL ÁRBOL DE DECISIÓN:")
print("-" * 80)
print(reglas)
print()

with open('c:\\Users\\diego.sanchez\\Documents\\l\\reglas_decision.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("REGLAS DE DECISIÓN DEL MODELO DE APROBACIÓN DE CRÉDITOS\n")
    f.write("=" * 80 + "\n\n")
    f.write(reglas)
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("IMPORTANCIA DE VARIABLES:\n")
    f.write("-" * 80 + "\n")
    for i in indices:
        f.write(f"{nombres_features[i]:<30} : {importancias[i]:.4f}\n")

print("Reglas guardadas en: reglas_decision.txt")
print()

# Variables más importantes
print("=" * 80)
print("VARIABLES MÁS IMPORTANTES EN LA DECISIÓN:")
print("-" * 80)

for i, idx in enumerate(indices[:5], 1):
    print(f"{i}. {nombres_features[idx]:<30} -> {importancias[idx]*100:.2f}%")
print()

# Función de simulación para nuevos clientes
print("=" * 80)
print("PASO 8: Función de simulación lista para usar")
print("-" * 80)
print()

def evaluar_nuevo_cliente(edad, ingresos_mensuales, deuda_total, score_crediticio,
                         antiguedad_credito_anos, num_cuentas):
    """
    Evalúa si un nuevo cliente debe ser aprobado para un crédito.
    
    Parámetros:
    -----------
    edad : int - Edad del solicitante (años)
    ingresos_mensuales : float - Ingresos mensuales en dólares
    deuda_total : float - Deuda total actual en dólares
    score_crediticio : int - Puntaje crediticio (300-850)
    antiguedad_credito_anos : int - Años de historial crediticio
    num_cuentas : int - Número de cuentas bancarias
    
    Retorna:
    --------
    dict : Diccionario con la decisión y probabilidades
    """
    ratio_deuda_ingreso = deuda_total / ingresos_mensuales if ingresos_mensuales > 0 else 999
    
    nuevo_cliente = pd.DataFrame({
        'Edad': [edad],
        'Ingresos_Mensuales': [ingresos_mensuales],
        'Deuda_Total': [deuda_total],
        'Score_Crediticio': [score_crediticio],
        'Antiguedad_Credito_Anos': [antiguedad_credito_anos],
        'Num_Cuentas': [num_cuentas],
        'Ratio_Deuda_Ingreso': [ratio_deuda_ingreso]
    })
    
    prediccion = modelo.predict(nuevo_cliente)[0]
    probabilidades = modelo.predict_proba(nuevo_cliente)[0]
    
    decision = "APROBADO" if prediccion == 1 else "RECHAZADO"
    prob_rechazo = probabilidades[0] * 100
    prob_aprobacion = probabilidades[1] * 100
    
    print("=" * 80)
    print("EVALUACIÓN DE NUEVO CLIENTE")
    print("=" * 80)
    print(f"DATOS DEL SOLICITANTE:")
    print(f"   - Edad: {edad} años")
    print(f"   - Ingresos Mensuales: ${ingresos_mensuales:,.2f}")
    print(f"   - Deuda Total: ${deuda_total:,.2f}")
    print(f"   - Score Crediticio: {score_crediticio} puntos")
    print(f"   - Antigüedad Crediticia: {antiguedad_credito_anos} años")
    print(f"   - Número de Cuentas: {num_cuentas}")
    print(f"   - Ratio Deuda/Ingreso: {ratio_deuda_ingreso:.2f}x")
    print()
    print(f"DECISIÓN DEL MODELO: {decision}")
    print()
    print(f"PROBABILIDADES:")
    print(f"   - Probabilidad de Rechazo: {prob_rechazo:.2f}%")
    print(f"   - Probabilidad de Aprobación: {prob_aprobacion:.2f}%")
    print()
    
    print(f"ANÁLISIS DE FACTORES:")
    factores_positivos = []
    factores_negativos = []
    
    if score_crediticio >= 700:
        factores_positivos.append(f"Excelente score crediticio ({score_crediticio})")
    elif score_crediticio < 600:
        factores_negativos.append(f"Score crediticio bajo ({score_crediticio})")
    
    if ratio_deuda_ingreso < 1.5:
        factores_positivos.append(f"Ratio deuda/ingreso saludable ({ratio_deuda_ingreso:.2f}x)")
    elif ratio_deuda_ingreso > 3:
        factores_negativos.append(f"Ratio deuda/ingreso muy alto ({ratio_deuda_ingreso:.2f}x)")
    
    if ingresos_mensuales > 2000:
        factores_positivos.append(f"Buenos ingresos (${ingresos_mensuales:,.2f})")
    elif ingresos_mensuales < 1000:
        factores_negativos.append(f"Ingresos bajos (${ingresos_mensuales:,.2f})")
    
    if antiguedad_credito_anos > 5:
        factores_positivos.append(f"Buen historial crediticio ({antiguedad_credito_anos} años)")
    
    if factores_positivos:
        print("   Factores a favor:")
        for factor in factores_positivos:
            print(f"      + {factor}")
    
    if factores_negativos:
        print("   Factores en contra:")
        for factor in factores_negativos:
            print(f"      - {factor}")
    
    print("=" * 80)
    print()
    
    return {
        'decision': prediccion,
        'decision_texto': decision,
        'prob_rechazo': prob_rechazo,
        'prob_aprobacion': prob_aprobacion,
        'ratio_deuda_ingreso': ratio_deuda_ingreso
    }

# Ejemplos de simulación
print("=" * 80)
print("EJEMPLOS DE SIMULACIÓN")
print("=" * 80)
print()

print("EJEMPLO 1: Cliente con buen perfil")
print("-" * 80)
resultado1 = evaluar_nuevo_cliente(
    edad=35,
    ingresos_mensuales=3000,
    deuda_total=2000,
    score_crediticio=750,
    antiguedad_credito_anos=10,
    num_cuentas=3
)

print("\nEJEMPLO 2: Cliente con perfil medio")
print("-" * 80)
resultado2 = evaluar_nuevo_cliente(
    edad=28,
    ingresos_mensuales=1500,
    deuda_total=2000,
    score_crediticio=650,
    antiguedad_credito_anos=4,
    num_cuentas=2
)

print("\nEJEMPLO 3: Cliente con perfil de alto riesgo")
print("-" * 80)
resultado3 = evaluar_nuevo_cliente(
    edad=22,
    ingresos_mensuales=800,
    deuda_total=4000,
    score_crediticio=520,
    antiguedad_credito_anos=1,
    num_cuentas=1
)

print("\nEJEMPLO 4: Tu caso de prueba")
print("-" * 80)
resultado4 = evaluar_nuevo_cliente(
    edad=30,
    ingresos_mensuales=1000,
    deuda_total=500,
    score_crediticio=680,
    antiguedad_credito_anos=5,
    num_cuentas=2
)

# Conclusión y análisis de negocio
print()
print("=" * 80)
print("CONCLUSIÓN Y ANÁLISIS DE NEGOCIO")
print("=" * 80)
print()
print("CÓMO AYUDA ESTE MODELO A UN BANCO A REDUCIR PÉRDIDAS")
print("-" * 80)
print()
print("1. AUTOMATIZACIÓN DE DECISIONES:")
print("   - El modelo puede evaluar miles de solicitudes en segundos")
print("   - Reduce el tiempo de respuesta de días a minutos")
print("   - Elimina sesgos humanos en la toma de decisiones")
print()
print("2. REDUCCIÓN DE MOROSIDAD:")
print(f"   - Precisión del modelo: {accuracy_test*100:.2f}%")
print(f"   - De cada 100 créditos otorgados, {int(precision*100)} son a clientes confiables")
print(f"   - Falsos positivos minimizados: {falsos_positivos} casos en {len(y_test)} evaluados")
print()
print("3. IDENTIFICACIÓN DE FACTORES CLAVE:")
print("   Las variables más importantes son:")
for i, idx in enumerate(indices[:3], 1):
    print(f"   {i}. {nombres_features[idx]} ({importancias[idx]*100:.1f}% de importancia)")
print()
print("4. TRANSPARENCIA Y EXPLICABILIDAD:")
print("   - El árbol de decisión muestra reglas claras (If-Then)")
print("   - Los auditores pueden entender por qué se rechazó/aprobó un crédito")
print("   - Cumple con regulaciones de transparencia bancaria")
print()
print("5. IMPACTO ECONÓMICO ESTIMADO:")
print(f"   - Si el banco procesa 10,000 solicitudes al mes:")
print(f"   - Y el modelo tiene {accuracy_test*100:.1f}% de precisión:")
print(f"   - Evita aprobar ~{int((1-precision)*10000*0.5)} créditos riesgosos")
print(f"   - Si cada crédito promedio es $5,000:")
print(f"   - Ahorro mensual potencial: ${int((1-precision)*10000*0.5*5000):,}")
print()
print("6. MEJORA CONTINUA:")
print("   - El modelo puede reentrenarse con datos reales")
print("   - Se adapta a cambios en el comportamiento del mercado")
print("   - Puede incorporar nuevas variables según necesidad")
print()
print("=" * 80)
print("PROYECTO COMPLETADO EXITOSAMENTE")
print("=" * 80)
print()
print("Archivos generados:")
print("   - evaluacion_modelo.png (Gráficos de evaluación)")
print("   - arbol_decision.png (Visualización del árbol)")
print("   - reglas_decision.txt (Reglas en formato texto)")
print()
print("Puedes usar la función evaluar_nuevo_cliente() para probar nuevos casos.")
print("=" * 80)
