# An Explainable AI Framework for Multimodal Big Data Analytics in Smart Transportation Systems

## Project Overview

**An Explainable AI Framework for Multimodal Big Data Analytics in Smart Transportation Systems** is an integrated Big Data Analytics and Explainable Artificial Intelligence (XAI) framework for analyzing transportation, road-safety, and climate/weather data.

The framework combines multiple transportation-related data sources, including **public transport feeds, road accident records, and climate/weather data**, to identify transportation patterns, road safety risks, environmental relationships, and key factors influencing transportation outcomes.

Machine Learning models are used for prediction, while Explainable AI techniques such as **SHAP** and **LIME** provide understandable explanations of model predictions.

The project aims to support safer transportation planning, better mobility, and informed decision-making through data-driven and interpretable analytics.

---

## Problem Statement

Modern smart cities generate large volumes of heterogeneous transportation data from public transport services, road accident reports, and meteorological records.

Existing transportation analytics often analyze these data sources independently, making it difficult to identify relationships between transportation patterns, road safety, and environmental conditions.

Furthermore, many machine learning models operate as black boxes, making it difficult to understand the factors influencing their predictions.

Therefore, this project proposes an integrated **Big Data Analytics and Explainable AI framework** that analyzes transportation and road-safety data while providing understandable insights for safer and smarter transportation planning.

---

## Objectives

* Develop an integrated data analytics framework for analyzing public transport, road accident, and climate/weather datasets.
* Identify transportation, road-safety, and environmental trends, patterns, relationships, and risk factors through data analysis and visualization.
* Integrate **SHAP and LIME** techniques to predict transportation and road-safety outcomes and provide meaningful explanations of model predictions.
* Evaluate model performance and explainability to identify important factors influencing transportation and road-safety predictions.
* Support informed transportation and road-safety decision-making through interpretable analytics.

---

## Data Sources

The proposed framework uses three major categories of datasets:

### 1. Public Transport Data

Public transportation feeds are used to analyze transportation operations, movement patterns, and related transportation characteristics.

### 2. Road Accident Data

Road accident records, including **Road Accidents in India (2023)**, are used to analyze accident patterns and road-safety risks.

### 3. Climate and Weather Data

**Climate in India Daily Weather Data (2000–2024)** is used to study the relationship between weather and transportation or road-safety conditions.

The project documentation identifies these three datasets as the benchmark data sources for the proposed framework.

---

## Proposed Methodology

The project follows a modular data analytics and Explainable AI pipeline.

```text
Multi-Source Transportation & Weather Data
                │
                ▼
      Data Preprocessing
   ┌────────────────────────┐
   │ Missing Value Handling │
   │ Encoding               │
   │ Scaling                │
   │ Temporal Features      │
   └────────────────────────┘
                │
                ▼
   Exploratory Data Analysis
   ┌────────────────────────┐
   │ Correlation Heatmaps   │
   │ Boxplots               │
   │ Missing Value Analysis │
   │ Spatial Analysis       │
   └────────────────────────┘
                │
                ▼
       Machine Learning
   ┌────────────────────────┐
   │ Decision Tree          │
   │ Random Forest          │
   │ XGBoost                │
   │ CatBoost               │
   │ Logistic Regression    │
   └────────────────────────┘
                │
                ▼
     Prediction & Evaluation
                │
                ▼
     Explainable AI (XAI)
   ┌────────────────────────┐
   │ SHAP                   │
   │ LIME                   │
   └────────────────────────┘
                │
                ▼
 Interactive Transportation
          Dashboard
```

---

## Machine Learning Models

The proposed framework will evaluate multiple machine learning algorithms:

* Decision Tree
* Random Forest
* XGBoost
* CatBoost
* Logistic Regression

The models can be applied to transportation prediction and classification tasks such as:

* Transportation delays
* Transportation volume
* Accident severity
* Transportation risk level

The project plan specifies regression and classification tasks for transportation and road-safety outcomes.

---

## Explainable AI

A major component of this project is the integration of **Explainable Artificial Intelligence (XAI)**.

### SHAP

**SHAP (SHapley Additive exPlanations)** will be used to understand how individual features contribute to model predictions.

Planned SHAP visualizations include:

* Summary plots
* Dependency plots
* Force plots
* Global feature importance

### LIME

**LIME (Local Interpretable Model-Agnostic Explanations)** will be used to explain individual predictions and identify the factors contributing to a particular transportation or road-safety prediction.

---

## Risk Assessment

The framework will categorize predicted transportation risks into actionable levels:

```text
Low Risk
   │
   ▼
Medium Risk
   │
   ▼
High Risk
```

For each risk level, the system will analyze the major environmental, transportation, and operational factors contributing to the prediction.

---

## Model Evaluation

Different evaluation metrics will be used depending on the prediction task.

### Regression

* MAE
* RMSE
* R²

### Classification

* Accuracy
* F1-Score
* ROC-AUC

Model performance and explainability will be evaluated together to identify suitable approaches for transportation decision-making.

---

## Interactive Dashboard

The proposed system will include an interactive transportation analytics dashboard using **Streamlit or Power BI**.

The dashboard is expected to provide:

* Transportation analytics
* Risk-level visualization
* Risk hotspot mapping
* Prediction results
* SHAP explanations
* LIME explanations
* Feature importance
* Environmental and operational insights

---

## Research Gaps Addressed

The project addresses the following research gaps:

### Gap 1 – Lack of Integrated Data Analysis

Existing studies often analyze public transport, road accident, and climate/weather data separately.

### Gap 2 – Limited Identification of Combined Risk Factors

There is limited focus on identifying transportation, road-safety, and environmental patterns together.

### Gap 3 – Lack of Explainable AI Integration

Many transportation AI models focus primarily on prediction accuracy and may operate as black boxes.

### Gap 4 – Limited Evaluation of Prediction and Explainability

Existing research gives less attention to evaluating both model performance and explainability together.

---

## Innovation

The main innovation of the project is a **Unified Big Data and Explainable AI Framework** that integrates:

* Multimodal transportation data
* Road accident data
* Climate/weather data
* Big Data Analytics
* Machine Learning
* Explainable AI
* Risk assessment
* Interactive visualization

The framework aims to explain the environmental and operational factors influencing transportation risk predictions.

---

## Project Structure

```text
KLH-CSE-2026-27-2420030677-Explainable-AI-for-Smart-Transportation-Analytics
│
├── data/
│   └── README.md
│
├── docs/
│   └── README.md
│
├── reports/
│   └── README.md
│
├── results/
│   └── README.md
│
├── src/
│   └── README.md
│
├── .gitignore
└── README.md
```

### Folder Description

| Folder     | Description                                                                     |
| ---------- | ------------------------------------------------------------------------------- |
| `data/`    | Transportation, road accident, climate, and weather datasets                    |
| `docs/`    | Project documentation, diagrams, literature survey, and related documents       |
| `reports/` | Review reports, analysis reports, and project reports                           |
| `results/` | Model outputs, evaluation results, graphs, and visualizations                   |
| `src/`     | Source code, preprocessing, machine learning, XAI implementation, and notebooks |

---

## Project Phases

### Phase 1 – Literature Review & Gap Finalization

Review recent research papers and finalize the research problem and gaps.

### Phase 2 – Data Ingestion & Integration

Collect and integrate public transport, road accident, and climate/weather datasets.

### Phase 3 – Exploratory Data Analysis & Preprocessing

Perform data quality analysis, missing-value handling, outlier detection, correlation analysis, encoding, scaling, and train-test splitting.

### Phase 4 – Predictive Machine Learning Modeling

Train and compare Decision Tree, Random Forest, XGBoost, CatBoost, and Logistic Regression models.

### Phase 5 – Explainable AI Implementation

Apply SHAP and LIME to explain global model behavior and individual predictions.

### Phase 6 – Multi-Tier Risk Assessment

Categorize transportation predictions into Low, Medium, and High Risk levels.

### Phase 7 – Interactive Dashboard Development

Develop an interactive dashboard with risk visualization, prediction controls, and XAI visualizations.

### Phase 8 – Model Validation & Documentation

Compare model performance, evaluate explainability, document findings, and prepare the final project report.

---

## Technologies Used

### Programming

* Python

### Data Analytics

* Pandas
* NumPy
* SciPy

### Machine Learning

* Scikit-learn
* XGBoost
* CatBoost

### Explainable AI

* SHAP
* LIME

### Big Data Processing

* Pandas
* Dask
* PySpark

### Visualization & Dashboard

* Matplotlib
* Streamlit
* Power BI

### Development & Version Control

* Jupyter Notebook
* Google Colab
* VS Code
* Git
* GitHub

---

## Feasibility

The project is designed using an open-source software stack and publicly accessible datasets.

* **Dataset:** Publicly accessible transportation and environmental datasets
* **Programming:** Python data science ecosystem
* **ML & XAI:** Well-supported open-source libraries
* **Big Data:** Pandas/Dask with PySpark scalability
* **Visualization:** Streamlit or Power BI
* **Hardware:** Standard laptop or free Google Colab environments
* **Cost:** Open-source software and publicly accessible data

---

## Expected Outcomes

The project is expected to:

* Analyze multimodal transportation data.
* Identify transportation, road-safety, and environmental patterns.
* Predict transportation and road-safety outcomes.
* Identify important risk factors.
* Explain model predictions using SHAP and LIME.
* Categorize transportation risks into actionable levels.
* Provide interactive visual insights through a dashboard.
* Support safer transportation planning and informed decision-making.

---

## Team Details

**Batch:** 12

| Student ID | Name                  |
| ---------- | --------------------- |
| 2420030554 | Y. Paavana Pranavasri |
| 2420030394 | K. Bhavya             |
| 2420030416 | A. Divya Sree         |
| 2420030677 | J. Sree Vardhini      |

---

## Project Status

🚧 **Project Under Development**

The repository will be updated progressively with datasets, source code, documentation, experiments, model results, visualizations, and final project reports.

---

## References

1. Hassan et al. (2025), *Application of machine learning in intelligent transport systems: A comprehensive review and bibliometric analysis*, Discover Civil Engineering, Springer.
2. Akin, Canbay & Sagiroglu (2025), *A novel geo-independent and privacy-preserved traffic speed prediction framework based on deep learning for intelligent transportation systems*, The Journal of Supercomputing, Springer.
3. Rocco di Torrepadula et al. (2025), *Fedflow: A personalized federated learning framework for passenger flow prediction*, Machine Learning, Springer.
4. Alam et al. (2025), *A comparative study of machine learning models for taxi-demand prediction using a big data framework*, Public Transport, Springer.
5. Leinonen et al. (2025), *Long Short-Term Memory-Based Traffic Prediction Using Multi-Source Data*, International Journal of Intelligent Transportation Systems Research, Springer.
6. Abdulrashid et al. (2025), *An interpretable machine learning framework for enhancing road transportation safety*, Transportation Research Part E.
7. Abdulrashid et al. (2025), *Transport behavior and government interventions in pandemics: A hybrid explainable machine learning for road safety*, Transportation Research Part E.
8. Chen et al. (2026), *From Prediction to Prevention: Using Text Mining and Explainable Machine Learning for Urban Bus Accident Analytics*, Risk Analysis.
9. Abid et al. (2026), *Improving road safety in smart cities using machine learning techniques*, Scientific Reports.
10. Pathak & Chatterjee (2026), *Artificial intelligence, machine learning, and deep learning in intelligent transportation systems: A review for smart cities and sustainable mobility*, International Journal of Applied Resilience and Sustainability.

