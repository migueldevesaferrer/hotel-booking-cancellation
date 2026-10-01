# Predicción de cancelaciones hoteleras

Proyecto de machine learning para estimar, en el momento de crear o confirmar una reserva, la probabilidad de que esta sea cancelada. El propósito es apoyar la priorización de acciones preventivas de Revenue Management sin emplear información que solo se conoce después de la reserva.

> Estado de los resultados: las métricas de la comparación de modelos que aparecen en este README proceden de la salida almacenada en el notebook. Las celdas posteriores de selección de umbral no conservaron su salida al guardar el notebook; por tanto, sus métricas no se presentan como resultados finales hasta reproducir la ejecución completa.

## 1. Dataset

El proyecto utiliza **Hotel Booking Demand**, publicado por Antonio, Almeida y Nunes (2019). El archivo de trabajo local se denomina `reservas_hoteleras.csv` y contiene reservas de hoteles urbanos y vacacionales.

- Variable objetivo: `is_canceled` (`1` si la reserva se cancela; `0` en caso contrario).
- Auditoría inicial: 119.390 observaciones, con una tasa de cancelación de **37,04 %**.
- Nulos relevantes: `company` (94,31 %), `agent` (13,69 %) y `country` (0,41 %).
- Se conservan los duplicados completos: el dataset no dispone de un identificador único de reserva, por lo que dos filas iguales pueden ser reservas distintas.

Referencia: Antonio, N., de Almeida, A. y Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126

## 2. Objetivos del proyecto

1. Estimar la probabilidad de cancelación con información disponible al reservar.
2. Comparar modelos interpretables y no lineales con una validación respetuosa con el orden temporal.
3. Traducir probabilidades en acciones de negocio mediante un umbral basado en costes.
4. Explicar las señales que influyen en cada predicción, sin confundir asociación con causalidad.

## 3. Stack tecnológico

- Python 3
- `pandas` y `numpy` para manipulación de datos
- `matplotlib` y `seaborn` para visualización
- `scikit-learn` para preprocesamiento, pipelines y modelos base
- `xgboost` y `optuna` para boosting y optimización de hiperparámetros
- `shap` para interpretabilidad global y local
- JupyterLab para la ejecución del análisis

Las versiones necesarias están recogidas en `requirements.txt`.

## 4. EDA — Insights principales

El análisis exploratorio se orienta a preguntas de negocio y muestra tanto tasas como volúmenes para evitar conclusiones sobre grupos poco representativos.

- **Tipo de hotel:** la tasa de cancelación observada es mayor en City Hotel (30,04 %) que en Resort Hotel (23,48 %).
- **Antelación de reserva:** el riesgo aumenta con el `lead_time`; pasa de 8,43 % para reservas de 0-7 días a 40,88 % para reservas con más de 365 días de antelación.
- **Estacionalidad:** julio y agosto concentran las tasas observadas más altas, 31,80 % y 32,18 % respectivamente.
- **Canal y segmento:** Online TA registra 35,35 % de cancelación; TA/TO, 30,97 %. Estos resultados deben leerse junto con el volumen de cada categoría.
- **ADR:** el ADR medio es mayor entre las reservas canceladas (117,77) que entre las no canceladas (102,00). Es una asociación, no una relación causal.
- **Categorías residuales:** los valores `Undefined` en `market_segment` y `distribution_channel` se mantienen como categoría propia. Su frecuencia es demasiado baja para derivar conclusiones operativas.

## 5. Variables excluidas por data leakage

La predicción se sitúa en el momento de creación o confirmación de la reserva. Se excluyen variables que pueden incorporar información posterior o derivada del desenlace:

| Variable | Motivo de exclusión |
| --- | --- |
| `reservation_status` | Es el estado final de la reserva y revela el objetivo. |
| `reservation_status_date` | Está asociada al estado final. |
| `booking_changes` | Puede acumular cambios realizados después de crear la reserva. |
| `assigned_room_type` | Puede reflejar una asignación realizada posteriormente. |
| `days_in_waiting_list` | Su disponibilidad depende del instante exacto de predicción; se adopta un criterio conservador. |
| `lead_time_group_eda` | Es una variable creada solo para exploración y no forma parte del pipeline. |

La discrepancia entre habitación reservada y asignada aparece en el 12,49 % de las reservas, lo que refuerza la exclusión de `assigned_room_type`.

## 6. Feature Engineering

Se añaden variables sencillas, disponibles a partir de datos de la propia reserva:

- `total_guests`: suma de adultos, niños y bebés.
- `total_nights`: suma de noches de fin de semana y entre semana.
- `has_children`: indicador de presencia de niños o bebés.

Tras las decisiones de limpieza, selección y creación de variables, el conjunto de modelización contiene 87.396 observaciones y 28 columnas, incluida la variable objetivo.

## 7. Modelos entrenados

El preprocesamiento se encapsula en un `Pipeline` con `ColumnTransformer`:

- Variables numéricas: imputación por mediana y escalado estándar.
- Variables categóricas: imputación por moda y One-Hot Encoding con categorías desconocidas ignoradas.

Se comparan los siguientes modelos:

1. **Regresión logística:** baseline interpretable.
2. **Regresión logística balanceada:** prioriza recall mediante ponderación de clases.
3. **Random Forest:** modelo no lineal con 300 árboles.
4. **XGBoost optimizado:** búsqueda de hiperparámetros con Optuna y validación temporal dentro de entrenamiento.

La partición es temporal: 75 % de entrenamiento (65.547 filas) y 25 % de test (21.849 filas). El test se mantiene separado de la optimización.

## 8. Resultados

### Comparación guardada en el notebook

La siguiente tabla corresponde exactamente a la salida persistida de la comparación de modelos. Todos se evalúan sobre el test temporal con umbral **0,50**.

| Modelo | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regresión logística | 0,8107 | 0,6897 | 0,7475 | 0,6420 | 0,6076 | 0,6243 |
| Regresión logística balanceada | 0,8100 | 0,6817 | 0,6441 | 0,4917 | **0,9111** | **0,6387** |
| Random Forest | 0,8159 | 0,6664 | 0,7377 | **0,6868** | 0,4417 | 0,5377 |
| XGBoost optimizado | **0,8303** | **0,6962** | **0,7495** | 0,6784 | 0,5221 | 0,5901 |

XGBoost obtiene la mejor capacidad de discriminación, medida por ROC-AUC y PR-AUC. La regresión logística balanceada maximiza el recall a costa de una precisión menor, por lo que genera más intervenciones sobre reservas que finalmente no se cancelarían.

### Selección del umbral de negocio

El notebook implementa una selección de umbral en validación interna con una hipótesis ilustrativa de coste `FN = 3 × FP`, y también compara XGBoost con umbral 0,15. Sin embargo, las salidas de esas celdas no están guardadas en el archivo publicado. Para mantener trazabilidad, no se incluyen aquí cifras de esos dos escenarios. Deben incorporarse al README solo después de ejecutar el notebook completo y guardar sus resultados.

## 9. Interpretabilidad (SHAP)

La interpretabilidad se usa para entender el comportamiento del modelo, no para atribuir causalidad.

- **Importancia global:** XGBoost ordena las variables transformadas según su contribución al modelo.
- **SHAP summary plot:** resume el efecto de las variables sobre una muestra reproducible del test.
- **Explicaciones locales:** se seleccionan dinámicamente las reservas de mayor y menor riesgo predicho y se muestran las variables que empujan la predicción en cada dirección.

La interpretación de variables categóricas codificadas con One-Hot requiere cuidado: que una categoría tenga valor `0` significa que no está presente en esa reserva, no que el cliente pertenezca a una categoría opuesta.

## 10. Conclusiones de negocio

- El modelo permite priorizar reservas con mayor probabilidad de cancelación antes de la llegada.
- La elección del umbral debe basarse en costes reales, capacidad operativa y valor de cada intervención; 0,50 es solo una referencia estadística.
- Una estrategia operativa razonable es definir bandas de riesgo: seguimiento habitual para riesgo bajo, recordatorios para riesgo medio y revisión proactiva para riesgo alto.
- Antes de producción se debe confirmar con negocio la disponibilidad de cada variable, calibrar probabilidades en periodos posteriores y validar el impacto económico con un piloto controlado.

## 11. Estructura del repositorio

```text
hotel-booking-cancellation/
|-- data/
|   |-- raw/                 # Dataset local, excluido de Git
|   |-- processed/           # Datos derivados, excluidos de Git
|   `-- README.md
|-- models/                  # Artefactos locales, excluidos de Git
|-- notebooks/
|   `-- hotel_booking_cancellation_final.ipynb
|-- reports/
|   `-- figures/             # Figuras publicables
|-- src/                     # Código reutilizable futuro
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## 12. Cómo ejecutar el proyecto

```powershell
git clone https://github.com/migueldevesaferrer/hotel-booking-cancellation.git
cd hotel-booking-cancellation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Antes de ejecutar el notebook, coloca `reservas_hoteleras.csv` en `data/raw/`. El dataset está excluido de Git por sus condiciones de uso y para evitar versionar archivos pesados. Abre `notebooks/hotel_booking_cancellation_final.ipynb` y ejecútalo de arriba abajo.
