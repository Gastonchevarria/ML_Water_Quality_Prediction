# ML Water Quality Prediction — Safe Drinking Water Modeling 💧🧪

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Dataset: Water Potability Benchmark](https://img.shields.io/badge/Dataset-Kaggle%20%2F%20WHO%20Standards-10B981.svg)]()
[![Method: QDA + PyCaret AutoML](https://img.shields.io/badge/Methodology-QDA%20%2B%20PyCaret%20AutoML-orange.svg)]()
[![Domain: Public Health & Sanitation](https://img.shields.io/badge/Domain-SDG%206%20%7C%20Safe%20Drinking%20Water-0284C7.svg)]()

> **Machine Learning para la Evaluación de Potabilidad y Seguridad Hídrica**  
> *Predicción Supervisada de Aptitud para Consumo Humano basada en Estándares OMS/EPA mediante Análisis Discriminante Cuadrático (QDA) y Comparación de Modelos AutoML.*

---

## 👨‍🔬 Dirección Científica y Autoría

- **Investigador Principal:** **Ing. Químico Gastón Chevarria**
- **Filiación Académica / Institucional:** Investigador Independiente · Especialista en Arquitectura de Sistemas de Inteligencia Artificial & GCP.  
  Formación de grado y posgrado: **Universidad Tecnológica Nacional (UTN)** · **Universidad Nacional de San Martín (UNSAM)** · **Instituto Data Science**.  
  GitHub: [@Gastonchevarria](https://github.com/Gastonchevarria) · LinkedIn: [/in/gastonchevarria-b9902a46](https://linkedin.com/in/gastonchevarria-b9902a46)
- **Ecosistema de Tecnologías Ambientales y Monitoreo Hídrico:**  
  Este proyecto forma parte de la línea de investigación aplicada en modelado ambiental y preservación de recursos hídricos desarrollada por Gastón Chevarria, en conjunción con:
  - [Clasificación de Calidad de Agua en la Franja Costera del Río de la Plata](https://github.com/Gastonchevarria/Water-Quality-Classification-of-the-coastal-zone-of-Buenos-Aires) (PCA + K-Means sobre series temporales 2013-2022).
  - [OceanEye — Evaluación Científica y Teledetección](https://github.com/Gastonchevarria/oceaneye-evaluation) (Monitoreo satelital Sentinel-1 SAR y soberanía en el Mar Argentino).

---

## 🧭 Resumen Ejecutivo y Marco Normativo

El acceso universal al agua potable segura constituye una de las metas cardinales del **Objetivo de Desarrollo Sostenible 6 (ODS 6)** de las Naciones Unidas y un derecho humano fundamental. En la gestión de plantas potabilizadoras y redes de distribución descentralizadas, la determinación oportuna de la inocuidad del agua requiere evaluar simultáneamente parámetros químicos, desinfectantes residuales, mineralización y subproductos de desinfección.

Este repositorio implementa un sistema predictivo de Machine Learning capaz de:
1. **Analizar 9 descriptores fisicoquímicos críticos** frente a los valores guía de la Organización Mundial de la Salud (OMS) y la Agencia de Protección Ambiental (EPA).
2. **Gestionar la asimetría de clases** (desbalance de muestras no potables vs. potables) mediante submuestreo estratificado (*downsampling balanceado*).
3. **Evaluar un espectro amplio de clasificadores** (Random Forest, LightGBM, SGD, Regresión Logística y Discriminante Lineal/Cuadrático) utilizando el entorno de AutoML **PyCaret**.
4. **Desplegar un modelo óptimo de Análisis Discriminante Cuadrático (QDA)** capaz de capturar las covarianzas no lineales y las interacciones cruzadas entre mineralización, pH y desinfección química.

---

## 📊 Especificación del Dataset y Parámetros Fisicoquímicos

El estudio utiliza un benchmark estandarizado de potabilidad de agua (`water_potability.csv`) con **3.276 muestras de agua** caracterizadas por 9 parámetros cuantitativos y una variable objetivo binaria:

* **Variable Objetivo:** `Potability`
  * `0`: No potable / Insegura para consumo humano.
  * `1`: Potable / Segura para consumo humano.

### Límites Normativos y Significado de los Parámetros

| Parámetro | Unidad | Rango Guía OMS / EPA | Importancia para la Salud Pública |
|---|---|---|---|
| **pH** | Unidades pH | 6.5 – 8.5 | Balance ácido-base. Valores extremos provocan corrosión de cañerías o incrustaciones e inhiben la desinfección por cloro. |
| **Dureza (*Hardness*)** | mg/L ($CaCO_3$) | 120 – 200 mg/L | Concentración de sales disueltas de calcio y magnesio. Afecta el sabor y la vida útil de equipos térmicos. |
| **Sólidos Totales Disueltos (*Solids*)** | ppm (mg/L) | < 500 – 1.000 mg/L | Mineralización global del agua. Altas concentraciones confieren palatabilidad salobre y problemas gastrointestinales. |
| **Cloraminas (*Chloramines*)** | ppm (mg/L) | < 4.0 mg/L | Desinfectante residual secundario generado por combinación de cloro y amoníaco para controlar recrecimiento biológico. |
| **Sulfatos (*Sulfate*)** | mg/L | < 250 – 500 mg/L | Sales minerales naturales del suelo y rocas. Concentraciones elevadas causan efectos laxantes y sabor amargo. |
| **Conductividad (*Conductivity*)** | $\mu\text{S/cm}$ | < 400 – 500 $\mu\text{S/cm}$ | Conducción iónica de la solución; directamente proporcional a los sólidos iónicos disueltos. |
| **Carbono Orgánico Total (*Organic Carbon*)** | mg/L | < 2.0 – 4.0 mg/L (agua pura < 25 mg/L) | Indicador de materia orgánica en descomposición y precursor principal de subproductos carcinógenos. |
| **Trihalometanos (*Trihalomethanes - THMs*)** | $\mu\text{g/L}$ | < 80 – 100 $\mu\text{g/L}$ | Compuestos químicos volátiles (cloroformo, bromodiclorometano) formados por reacción del cloro con materia orgánica; carcinógenos potenciales. |
| **Turbidez (*Turbidity*)** | NTU | < 1.0 – 5.0 NTU | Presencia de partículas en suspensión que protegen a microorganismos patógenos frente a la luz ultravioleta y cloro. |

---

## 📐 Formulación Matemática y Pipeline de Modelado

### 1. Imputación Paramétrica y Balanceo de Clases
- Las observaciones faltantes en `ph`, `Sulfate` y `Trihalomethanes` se imputan mediante la **mediana condicional de la distribución marginal**, preservando la robustez frente a valores atípicos (*outliers*).
- Dada la asimetría original de la base ($61\%$ no potables vs. $39\%$ potables), se aplica una estrategia de **submuestreo aleatorio estratificado** para equilibrar el espacio muestral a proporciones idénticas ($50\% / 50\%$), evitando el sesgo de frontera hacia la clase mayoritaria.

### 2. Formulación del Clasificador QDA (Quadratic Discriminant Analysis)
A diferencia del Análisis Discriminante Lineal (LDA) que asume una matriz de covarianza idéntica $\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}$ para todas las clases, el **Análisis Discriminante Cuadrático (QDA)** modela matrices de covarianza independientes $\boldsymbol{\Sigma}_k$ para cada clase $k \in \{0, 1\}$.

Esto permite trazar una **superficie de decisión cuadrática no lineal** en el espacio $\mathbb{R}^9$:

$$P(Y = k \mid X = x) = \frac{\pi_k f_k(x)}{\sum_{l} \pi_l f_l(x)}$$

Donde la función de densidad condicional gaussiana multivariada es:

$$f_k(x) = \frac{1}{(2\pi)^{p/2} |\boldsymbol{\Sigma}_k|^{1/2}} \exp\left( -\frac{1}{2} (x - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (x - \boldsymbol{\mu}_k) \right)$$

Maximizando la función discriminante cuadrática:

$$\delta_k(x) = -\frac{1}{2} \ln |\boldsymbol{\Sigma}_k| - \frac{1}{2} (x - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (x - \boldsymbol{\mu}_k) + \ln \pi_k$$

Esta propiedad es esencial en química ambiental, donde las correlaciones entre dureza, sólidos y sulfatos difieren radicalmente entre matrices de agua tratada vs. contaminada.

---

## 📈 Rendimiento Experimental y Métricas de Validación

La evaluación del modelo se llevó a cabo sobre una partición independiente de prueba (*Holdout Test Set*) del **20%** ($N = 512$ muestras):

### Métricas Cuantitativas Globales

| Métrica | Valor Obtenido | Interpretación Técnica |
|---|---|---|
| **Exactitud (*Accuracy*)** | **60.94%** (0.61) | Concordancia global de clasificación sobre el dataset balanceado |
| **Precisión Clase Potable (Clase 1)** | **66.84%** (0.67) | De las muestras predichas como potables, 67% son efectivamente seguras |
| **Exhaustividad Clase Potable (*Recall*)** | **48.68%** (0.49) | Detección de muestras aptas sin comprometer el umbral de seguridad |
| **Precisión Clase No Potable (Clase 0)** | **57.37%** (0.57) | Identificación certera de agua insegura para consumo |
| **Exhaustividad Clase No Potable (Clase 0)** | **74.09%** (0.74) | **Alta sensibilidad en detección de contaminación**: captura el 74% del agua peligrosa |
| **F1-Score Ponderado** | **0.60** | Media armónica balanceada entre precisión y sensibilidad |

### Matriz de Confusión ($N = 512$)

```text
                  Predicción Clase 0      Predicción Clase 1
                   (No Potable)                (Potable)
Real Clase 0:          183                        64
Real Clase 1:          136                        129
```

> 🛡️ **Criterio de Seguridad en Salud Pública:**  
> En el monitoreo de agua de consumo, el costo de un **falso positivo** (declarar potable un agua contaminada) es severo. El modelo prioriza la especificidad y el recall de la clase no potable (**74.09% de acierto en detección de agua no potable**, minimizando vertidos contaminados desapercibidos).

---

## 💻 Estructura del Repositorio y Reproducción

```text
.
├── ML - Water Quality Prediction - Gaston.ipynb  # Notebook principal con EDA, PyCaret y QDA
├── water_potability.csv                           # Benchmark de 3.276 muestras de agua
└── README.md
```

### Reproducción Paso a Paso

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/Gastonchevarria/ML_Water_Quality_Prediction.git
   cd ML_Water_Quality_Prediction
   ```

2. Instalar el entorno de dependencias:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn pycaret lightgbm
   ```

3. Abrir y ejecutar el notebook:
   ```bash
   jupyter notebook "ML - Water Quality Prediction - Gaston.ipynb"
   ```

---

## 🔗 Ecosistema de Investigación y Proyectos Relacionados

- [Water Quality Classification of the Coastal Zone of Buenos Aires](https://github.com/Gastonchevarria/Water-Quality-Classification-of-the-coastal-zone-of-Buenos-Aires): Análisis no supervisado mediante PCA y K-Means de 643 muestras estacionales del Río de la Plata en colaboración con MAyDS y la red RIIGLO.
- [OceanEye — Evaluation & Scientific Verification Kit](https://github.com/Gastonchevarria/oceaneye-evaluation): Sistema satelital de teledetección marítima y soberanía pesquera mediante Sentinel-1 SAR C-Band y peritaje criptográfico Ed25519.

---

## 📄 Licencia

Distribuido bajo la Licencia **MIT**. Desarrollado con propósitos de investigación académica, bioinformática ambiental y desarrollo de herramientas de monitoreo sanitario abierto.
