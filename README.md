# 🕵️‍♂️ Detección de Reseñas Falsas en E-Commerce con PLN y Machine Learning

## 📝 Descripción del Proyecto
Este proyecto se centra en la implementación de modelos de Machine Learning y técnicas de Procesamiento de Lenguaje Natural (PLN) para clasificar automáticamente reseñas de productos en plataformas de comercio digital. El objetivo principal es distinguir de manera eficaz entre reseñas auténticas escritas por humanos y reseñas falsas generadas por computadora (bots), combatiendo la manipulación de la reputación en línea.

## 👥 Equipo de Trabajo
Desarrollado colaborativamente por:
* **Miguel Alejandro Pacheco Molina**
* **Mauricio Hamabiel Cortes Moreno**
* **Alejandro Maldonado López**
* **Jesús Armando Soria Martínez**

## 📊 Conjunto de Datos
* **Fuente:** Fake Reviews Dataset de Kaggle.
* **Volumen:** 40,000 reseñas de productos de la categoría "Home and Kitchen".
* **Distribución:** Conjunto perfectamente balanceado con 20,000 reseñas originales (humanas) y 20,000 reseñas falsas (generadas por computadora).

## ⚙️ Metodología y Tecnologías Aplicadas
* **Preprocesamiento Diferenciado:** Se diseñaron dos flujos de limpieza; uno estricto (eliminación total de ruido y caracteres especiales) optimizado para vectorización estadística, y uno ligero (conservando puntuación básica) para preservar el contexto semántico.
* **Vectorización de Texto:** * TF-IDF simple (Unigramas).
  * TF-IDF enriquecido (Unigramas + Bigramas con exclusión de términos extremos).
  * Embeddings Densos preentrenados GloVe (50 dimensiones).
* **Modelos Evaluados:** Regresión de Huber (adaptada para clasificación binaria), Naive Bayes (MultinomialNB), Máquinas de Vectores de Soporte (SVM con Kernel Lineal) y Perceptrón Multicapa (MLP).
* **Tecnologías:** Python, NLTK, Scikit-learn, Pandas, Matplotlib, Seaborn.

## 🏆 Resultados y Conclusiones Destacadas
Se evaluó el desempeño mediante validaciones cruzadas con diferentes proporciones de datos (90/10, 80/20, 70/30).

* **El Modelo Óptimo:** La **Regresión de Huber combinada con TF-IDF enriquecido (bigramas)** demostró ser el sistema más robusto. Logró un Accuracy superior al 92%, un F1-score de 0.9262 y un AUC de 0.9744 en la división 90/10.
* **Desempeño de SVM:** El modelo SVM con TF-IDF enriquecido también mostró un desempeño excepcional y gran estabilidad como clasificador puro.
* **Conclusión Analítica:** Los resultados evidenciaron que, para la tarea de diferenciar entre lenguaje humano y de bots, las características léxicas y de estilo capturadas por herramientas estadísticas (TF-IDF con bigramas) son considerablemente más determinantes y eficaces que el análisis semántico profundo proporcionado por embeddings densos como GloVe.
