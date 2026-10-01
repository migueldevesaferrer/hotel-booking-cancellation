# Predicción de cancelaciones hoteleras

Proyecto de machine learning para estimar, en el momento de confirmar una reserva, la probabilidad de que sea cancelada. El objetivo es apoyar la priorización de acciones preventivas de Revenue Management sin utilizar información que solo se conoce después de la reserva.

El analisis se basa en el dataset Hotel Booking Demand de Antonio, Almeida y Nunes (2019). La variable objetivo es `is_canceled`.

## Objetivo de negocio

Una predicción de riesgo permite ordenar las reservas por prioridad y aplicar acciones graduales, desde un seguimiento habitual para riesgo bajo hasta una revisión proactiva para riesgo alto. La decisión operativa no depende solo de la calidad estadística: el umbral debe incorporar el coste real de falsos positivos y falsos negativos.

## Resultados principales

La evaluación reserva el 25% más reciente de las observaciones como test temporal. La siguiente tabla recoge los resultados almacenados en la ejecución del notebook con umbral 0.50.

| Modelo | ROC-AUC | PR-AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regresión logística | 0.8107 | 0.6897 | 0.6420 | 0.6076 | 0.6243 |
| Regresión logística balanceada | 0.8100 | 0.6817 | 0.4917 | 0.9111 | 0.6387 |
| Random Forest | 0.8159 | 0.6664 | 0.6868 | 0.4417 | 0.5377 |
| XGBoost optimizado | **0.8303** | **0.6962** | 0.6784 | 0.5221 | 0.5901 |

XGBoost obtiene la mejor capacidad de discriminación. La regresión logística balanceada alcanza el mayor recall, con el coste de generar más intervenciones innecesarias. El notebook también ajusta un umbral mediante una hipótesis ilustrativa de coste `FN = 3 x FP`; debe recalibrarse con datos operativos reales antes de cualquier uso en producción.

## Metodologia

- Auditoría de calidad de datos y análisis exploratorio orientado a negocio.
- Exclusión de variables con posible fuga de información, como el estado final de la reserva y la habitación finalmente asignada.
- Creación de `total_guests`, `total_nights` y `has_children`.
- Partición temporal train/test y validación temporal en la optimización.
- Preprocesamiento encapsulado en `Pipeline` y `ColumnTransformer` para impedir fugas.
- Comparación de regresión logística, Random Forest y XGBoost optimizado con Optuna.
- Interpretabilidad global y local mediante importancias de XGBoost y SHAP.

## Estructura

```text
hotel-booking-cancellation/
|-- data/
|   |-- raw/                 # Dataset local, excluido de Git
|   |-- processed/           # Datos derivados, excluidos de Git
|   `-- README.md
|-- models/                  # Artefactos de modelo locales, excluidos de Git
|-- notebooks/
|   `-- hotel_booking_cancellation_final.ipynb
|-- reports/
|   `-- figures/             # Figuras que se quieran publicar
|-- src/                     # Codigo reutilizable cuando se extraiga del notebook
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Datos

El archivo de trabajo debe llamarse `reservas_hoteleras.csv` y ubicarse en `data/raw/`. No se versiona para evitar publicar datos sin revisar sus condiciones de uso. El notebook busca primero esta ruta y conserva compatibilidad con las rutas anteriores.

Referencia del dataset: Antonio, N., de Almeida, A. y Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126

## Instalación y ejecución

```powershell
git clone <URL_DEL_REPOSITORIO>
cd hotel-booking-cancellation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Abre `notebooks/hotel_booking_cancellation_final.ipynb` y ejecutalo de arriba abajo. El notebook requiere que el CSV este disponible en `data/raw/reservas_hoteleras.csv`.

## Limitaciones y siguientes pasos

La fecha de llegada se usa como aproximación temporal porque el dataset no incluye la fecha exacta de creación de la reserva. Antes de desplegar el enfoque, conviene validar con negocio la disponibilidad real de cada variable, estimar los costes de error, calibrar el umbral y medir el impacto económico de las acciones con un piloto controlado.
