# Introducción:
El código se divide en secciones claras con explicaciones detalladas para cada paso. Se incluyen ejemplos de cómo interpretar y visualizar los resultados del modelo.

## Descripción del código:
### 1. Importación de librerías:
- Se importan las librerías necesarias para la manipulación de datos (`pandas`), preprocesamiento (`numpy`, `sklearn.preprocessing`), creación de modelos (`sklearn.model_selection`, `sklearn.tree`) y evaluación de métricas (`sklearn.metrics`).
- Se importa la librería `SMOTE` para el balanceo de clases.
- Se importan las librerías para montar Google Drive (`google.colab`) y generar gráficos (`matplotlib`, `seaborn`).

### 2. Montaje de Google Drive:
Se monta Google Drive en el entorno de Colab para acceder a los archivos almacenados.

### 3. Carga del conjunto de datos:
Se carga el conjunto de datos CSV desde Google Drive en un DataFrame de `pandas`.

### 4. Selección de características:
Se seleccionan las columnas relevantes para el modelo ('amount', 'oldbalanceOrg', 'newbalanceOrig', 'type') como características y la columna objetivo ('isFraud').

### 5. Preprocesamiento de datos:
- Se aplica codificación one-hot a la columna 'type' para convertir las categorías en variables binarias.
- Se rellenan los valores faltantes utilizando `KNNImputer`.
- Se estandarizan las características numéricas con `StandardScaler`.
- Se balancean las clases del conjunto de datos utilizando `SMOTE`.

### 6. División de datos en entrenamiento y prueba:
Se divide el conjunto de datos en conjuntos de entrenamiento y prueba (80% para entrenamiento y 20% para prueba) para evaluar el rendimiento del modelo.

### 7. Entrenamiento del modelo base:
- Se entrena un modelo de árbol de decisión como modelo base.
- Se calcula el puntaje AUC-ROC en el conjunto de prueba para evaluar el rendimiento del modelo base.

### 8. Optimización de hiperparámetros (opcional):
- Si el puntaje base es inferior a 0.8, se realiza una optimización de hiperparámetros utilizando `GridSearchCV` para encontrar la mejor configuración para el modelo de árbol de decisión.
- Se reentrena el modelo con los mejores hiperparámetros encontrados y se evalúa su rendimiento.

### 9. Evaluación del modelo:
- Se calculan las siguientes métricas de rendimiento: exactitud, ROC AUC, informe de clasificación, matriz de confusión, precisión, recall, F1-score y AUC-PRC.
- Se visualizan las métricas utilizando gráficos de barras y mapas de calor para facilitar la interpretación.

### 10. Visualización de las reglas del árbol de decisión:
Se extraen las reglas del árbol de decisión entrenado y se muestran en un formato legible.

## Resultados del Modelo

### Exactitud (Accuracy):
La exactitud o precisión del modelo es aproximadamente 0.9975. Este valor indica que el modelo clasificó correctamente alrededor del 99.75% de las instancias en el conjunto de datos de prueba. Una exactitud cercana al 100% generalmente se considera excelente, ya que significa que el modelo está realizando predicciones precisas en la mayoría de los casos.

### ROC AUC:
El área bajo la curva ROC (ROC AUC, por sus siglas en inglés) es aproximadamente 0.9985. Esta métrica evalúa el rendimiento del modelo en un rango de umbrales de clasificación. Un valor de ROC AUC cercano a 1 indica que el modelo es capaz de distinguir entre las clases de una manera casi perfecta. Un valor de 0.9985 se considera extremadamente alto y representa un excelente desempeño del modelo.

En resumen, tanto la exactitud como el ROC AUC muestran un rendimiento excepcional del modelo. Una exactitud cercana al 99.75% y un ROC AUC cercano a 0.9985 sugieren que el modelo es capaz de clasificar correctamente la gran mayoría de las instancias y distinguir entre las clases de manera casi perfecta. Sin embargo, es importante tener en cuenta que estas métricas por sí solas no proporcionan una imagen completa del rendimiento del modelo. Sería recomendable analizar también otras métricas como la precisión (precision), el recall (sensibilidad), el F1-score, y la matriz de confusión para tener una evaluación más completa del modelo, especialmente si se trata de un problema con clases desbalanceadas.

### Clases desbalanceadas:
- La matriz muestra un claro desbalance entre las clases "No Fraude" y "Fraude". La clase "No Fraude" tiene muchas más instancias que la clase "Fraude".
- **Verdaderos Negativos (VN)**: El modelo clasificó correctamente 1,269,756 instancias como "No Fraude" cuando realmente no eran fraudes.
- **Falsos Negativos (FN)**: El modelo clasificó incorrectamente 1,580 instancias como "No Fraude" cuando en realidad eran fraudes. Estos son casos de fraudes que no fueron detectados por el modelo.
- **Falsos Positivos (FP)**: El modelo clasificó incorrectamente 1,755 instancias como "Fraude" cuando en realidad no lo eran. Estos son casos de no fraude que fueron identificados erróneamente como fraudes.
- **Verdaderos Positivos (VP)**: El modelo clasificó correctamente 633,231 instancias como "Fraude" cuando realmente eran fraudes.

#### Rendimiento en la clase "No Fraude":
El modelo tiene un buen rendimiento en la identificación de instancias no fraudulentas, con una tasa de acierto muy alta (1269756 / (1269756 + 1580) = 0.9987 o 99.87%).

#### Rendimiento en la clase "Fraude":
El modelo tiene un rendimiento relativamente bueno en la identificación de instancias fraudulentas, con una tasa de acierto de 633231 / (633231 + 1755) = 0.9973 o 99.73%.

### Impacto de los falsos negativos:
Los 1580 falsos negativos son casos de fraudes que no fueron detectados por el modelo, lo cual podría tener consecuencias significativas en términos de pérdidas financieras o de otro tipo.

### Impacto de los falsos positivos:
Los 1755 falsos positivos son casos de no fraude que fueron identificados erróneamente como fraudes. Esto podría generar costos adicionales en términos de investigaciones innecesarias o acciones incorrectas.

En general, el modelo parece tener un buen rendimiento en la identificación de ambas clases, aunque el desbalance de clases podría requerir ajustes adicionales. Sería importante evaluar el impacto y los costos asociados a los falsos negativos y falsos positivos en el contexto específico del problema para determinar si el rendimiento del modelo es aceptable o si se necesitan mejoras adicionales.

## Análisis de Métricas Adicionales:
Los valores de todas las variables son positivos, lo que indica un buen rendimiento del modelo.

### Precisión:
La precisión mide la proporción de predicciones correctas entre todas las predicciones realizadas. Un valor de 1.0 indica que todas las predicciones fueron correctas, mientras que un valor de 0.0 indica que todas las predicciones fueron incorrectas. En este caso, la precisión es de 0.999, lo que significa que el modelo realizó predicciones correctas en el 99.9% de los casos.

### Recall:
El recall mide la proporción de casos positivos que fueron identificados correctamente por el modelo. Un valor de 1.0 indica que todos los casos positivos fueron identificados correctamente, mientras que un valor de 0.0 indica que ningún caso positivo fue identificado correctamente. En este caso, el recall es de 0.998, lo que significa que el modelo identificó correctamente el 99.8% de los casos positivos.

### F1-score:
El f-score es una medida que combina la precisión y el recall en una sola métrica. Se calcula como la media armónica de la precisión y el recall. Un valor de 1.0 indica un rendimiento perfecto, mientras que un valor de 0.0 indica un rendimiento muy pobre. En este caso, el f-score es de 0.997, lo que significa que el modelo tuvo un buen rendimiento general.

### AUC-PRC:
El AUC-PRC (área bajo la curva de precisión-recuerdo) es una medida que evalúa el rendimiento del modelo en todos los umbrales de clasificación. Un valor de 1.0 indica un rendimiento perfecto, mientras que un valor de 0.0 indica un rendimiento muy pobre. En este caso, el AUC-PRC es de 0.996, lo que significa que el modelo tuvo un buen rendimiento en todos los umbrales de clasificación.

En general, el gráfico muestra que el modelo tuvo un buen rendimiento en todas las métricas evaluadas. La precisión, el recall, el f-score y el AUC-PRC son todos altos, lo que indica que el modelo fue capaz de identificar correctamente los casos positivos y negativos.

### Observaciones Adicionales:
- El valor de la precisión es ligeramente superior al valor del recall. Esto significa que el modelo es más propenso a clasificar correctamente los casos negativos que los casos positivos.
- El valor del f-score es muy cercano al valor de la precisión y el recall. Esto significa que el modelo tiene un buen equilibrio entre la precisión y el recall.
- El valor del AUC-PRC es muy alto. Esto significa que el modelo tiene un buen rendimiento en todos los umbrales de clasificación.
