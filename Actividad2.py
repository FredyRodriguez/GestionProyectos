import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

# Cargar los datos del archivo CSV
file_path = "Actividad2.csv"
df = pd.read_csv(file_path, sep=";")

# Seleccionar las columnas relevantes
relevant_columns = ['Gender', 'Education', 'Occupation', 'MaritalStatus', 'HomeOwnerFlag', 'NumberCarsOwned', 'NumberChildrenAtHome', 'TotalChildren', 'YearlyIncome', 'BikeBuyerText']
existing_columns = df.columns.intersection(relevant_columns)
df = df[existing_columns]

# Codificar las variables categóricas utilizando LabelEncoder
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

# Separar las características (X) y la variable objetivo (y)
X = df.drop('BikeBuyerText', axis=1)
y = df['BikeBuyerText']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de árbol de decisión
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred_dt = decision_tree_model.predict(X_test)


# Calcular la matriz de confusión
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

# Crear los nombres de las columnas y filas de la matriz de confusión
class_names = ['No Comprador de Bicicleta', 'Comprador de Bicicleta']

# Crear un DataFrame de pandas con la matriz de confusión y los nombres de columnas y filas
conf_matrix_df = pd.DataFrame(conf_matrix_dt, index=class_names, columns=class_names)

# Imprimir la matriz de confusión con nombres de columnas y filas
print("Matriz de confusión:")
print(conf_matrix_df)

# Explicar la matriz de confusión
print("\nExplicación de la matriz de confusión:")
print("- La columna 'No Comprador de Bicicleta' muestra las predicciones para la clase 'No Comprador de Bicicleta'.")
print("- La columna 'Comprador de Bicicleta' muestra las predicciones para la clase 'Comprador de Bicicleta'.")
print("- La fila 'No Comprador de Bicicleta' muestra las instancias reales de la clase 'No Comprador de Bicicleta'.")
print("- La fila 'Comprador de Bicicleta' muestra las instancias reales de la clase 'Comprador de Bicicleta'.")
print("- La diagonal principal muestra las predicciones correctas para cada clase.")
print("- Los valores fuera de la diagonal principal representan las predicciones incorrectas.")

# Calcular las métricas de evaluación
accuracy = accuracy_score(y_test, y_pred_dt)
precision = precision_score(y_test, y_pred_dt)
recall = recall_score(y_test, y_pred_dt)
f1 = f1_score(y_test, y_pred_dt)

print(f"\nAccuracy (Exactitud): {accuracy:.4f}")
print(f"Precision (Precisión) para la clase 'Comprador de Bicicleta': {precision:.4f}")
print(f"Recall (Sensibilidad) para la clase 'Comprador de Bicicleta': {recall:.4f}")
print(f"F1-score (Puntuación F1) para la clase 'Comprador de Bicicleta': {f1:.4f}")




# Obtener los nombres de las características utilizadas durante el entrenamiento
feature_names = X.columns

# Nuevos datos para realizar predicciones
new_data = pd.DataFrame({
    'Gender': ['M', 'F', 'M', 'F', 'M'],
    'Education': ['Bachelors', 'High School', 'Partial College', 'Graduate Degree', 'Bachelors'],
    'Occupation': ['Professional', 'Skilled Manual', 'Clerical', 'Management', 'Manual'],
    'MaritalStatus': ['M', 'S', 'M', 'S', 'M'],
    'HomeOwnerFlag': [1, 0, 1, 1, 0],
    'NumberCarsOwned': [2, 1, 2, 1, 0],
    'NumberChildrenAtHome': [2, 0, 1, 0, 2],
    'TotalChildren': [2, 0, 1, 0, 2],
    'YearlyIncome': [80000, 45000, 65000, 110000, 52000]
})

# Reordenar las columnas de los nuevos datos para que coincidan con las características utilizadas durante el entrenamiento
new_data = new_data[feature_names]

# Codificar las variables categóricas de los nuevos datos utilizando LabelEncoder
for column in ['Gender', 'Education', 'Occupation', 'MaritalStatus']:
    if column in label_encoders:
        new_data[column] = label_encoders[column].transform(new_data[column])

# Realizar predicciones en los nuevos datos
predictions = decision_tree_model.predict(new_data)

# Imprimir las predicciones con explicaciones
print("\nPredicciones:")
for i, prediction in enumerate(predictions, 1):
    if prediction == 1:
        print(f"Cliente {i}: Se predice que el cliente será un comprador de bicicleta.")
        print("  Este cliente tiene características similares a los compradores de bicicletas en nuestros datos históricos.")
        print("  Es probable que esté interesado en adquirir una bicicleta.")
    else:
        print(f"Cliente {i}: Se predice que el cliente no será un comprador de bicicleta.")
        print("  Las características de este cliente no coinciden con el perfil típico de un comprador de bicicleta.")
        print("  Es menos probable que esté interesado en adquirir una bicicleta en este momento.")
    print()
