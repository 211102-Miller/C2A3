import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

# Definir las cantidades de iteraciones y valor de K directamente como variables
iterations = 100  # Cantidad de iteraciones (generaciones)
folds = 5         # Valor de K para validación cruzada

# Cargar datos desde el archivo CSV
file_path = '211100.csv'
df_datos = pd.read_csv(file_path, delimiter=';')

# Corregir los nombres de columna
column_names = ['x1', 'x2', 'x3', 'y']
df_datos.columns = column_names

# Dividir los datos en conjuntos de entrenamiento y prueba
X = df_datos[['x1', 'x2', 'x3']].values
Y = df_datos['y'].values

# Inicializar listas para almacenar el MSE, los pesos y los valores del MSE por iteración
mse_values = []
weights = []            # Aquí se guardarán los pesos de cada partición
mse_per_iteration = []

# Realizar validación cruzada con k-fold
kf = KFold(n_splits=folds, shuffle=True, random_state=42)

best_partition = -1 
best_mse = float('inf')

lista_history = []

# Bucle para la validación cruzada
for i, (train_index, test_index) in enumerate(kf.split(X), 1):
    # Dividir el conjunto de datos en entrenamiento y prueba para esta partición
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    # Capa de salida única con activación lineal
    layer = tf.keras.layers.Dense(units=1, input_shape=(3,), activation='linear')
    model = tf.keras.Sequential([layer])

    # Compilar el modelo con el optimizador 'adam' y la función de pérdida 'mean_squared_error' para regresión lineal
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    # Entrenar el modelo y guardar el historial del entrenamiento
    history = model.fit(X_train, Y_train, epochs=iterations, verbose=0)
    lista_history.append(history.history['mse'])
    
    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcular el MSE entre las predicciones y las etiquetas reales
    mse = mean_squared_error(Y_test, y_pred)
    mse_values.append(mse)
    
    # Guardar los pesos del modelo en la lista 'weights'
    weights.append(model.get_weights()[0].flatten())
    
    # Guardar el MSE de esta partición si es el menor hasta ahora
    if mse < best_mse:
        best_mse = mse
        best_partition = i

# Convertir la lista de pesos en un DataFrame
weights_table = pd.DataFrame(weights, columns=['w1', 'w2', 'w3'])

# Crear una ventana de Tkinter para mostrar la tabla
table_window = tk.Tk()
table_window.title("Tabla de Resultados")
table_window.geometry("800x400")

# Crear el Treeview para mostrar la tabla
table_frame = ttk.Frame(table_window)
table_frame.pack(padx=10, pady=10)

tree = ttk.Treeview(table_frame, columns=['w1', 'w2', 'w3', 'Sesgo', 'MSE'])
tree.heading('#0', text='Partición')
tree.heading('w1', text='Peso w1')
tree.heading('w2', text='Peso w2')
tree.heading('w3', text='Peso w3')
tree.heading('Sesgo', text='Sesgo')
tree.heading('MSE', text='MSE')

# Insertar los datos en el Treeview
for i in range(folds):
    tree.insert('', 'end', text=f'Partición {i+1}', values=[weights_table.at[i, 'w1'], weights_table.at[i, 'w2'], weights_table.at[i, 'w3'], weights_table.at[i, 'w1'], mse_values[i]])

tree.pack()

# Mostrar la mejor partición
best_partition_label = tk.Label(table_window, text=f"La mejor partición: {best_partition}, MSE: {best_mse}")
best_partition_label.pack()

# Ejecutar la ventana de Tkinter para mostrar la tabla
table_window.mainloop()

# Gráfico de evolución
plt.figure()
for i in range(folds):
    plt.plot(lista_history[i], label=f'Partición {i+1}')
plt.xlabel('Época')
plt.ylabel('MSE')
plt.title('Evolución del MSE')
plt.legend()

# Mostrar todas las gráficas
plt.show()
