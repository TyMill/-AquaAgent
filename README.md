
# AquaAgent

[![PyPI version](https://img.shields.io/pypi/v/AquaAgent.svg)](https://pypi.org/project/AquaAgent/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15096947.svg)](https://doi.org/10.5281/zenodo.15098587)
[![Docs](https://img.shields.io/badge/docs-online-blue?logo=readthedocs&style=flat-square)](https://TyMill.github.io/-AquaAgent/)

**AquaAgent** is an autonomous and interactive AI agent for intelligent analysis and reporting of water quality data.

---

## 🔍 Features

- 📥 **Data Import**: CSV, Excel, JSON, SQL, MongoDB, APIs
- 🧼 **Cleaning & Imputation**: Mean, median, KNN, regression, autoencoder
- 📊 **Statistical Analysis**: Descriptive stats, ANOVA, time series decomposition
- 🤖 **ML Models**: Classification, regression, clustering, anomaly detection
- 🔁 **AutoML**: TPOT, AutoSklearn, Optuna optimization
- 🔎 **Interpretability**: SHAP, LIME
- 📈 **Visualization**: Heatmaps, PCA, t-SNE, interactive plots
- 🧾 **Reports**: HTML reports with insights and recommendations

---

## 🚀 Installation

```bash
pip install AquaAgent
```

Or from source:

```bash
git clone https://github.com/TyMill/AquaAgent.git
cd AquaAgent
pip install .
```

---

## 🧠 Usage Example

```bash
run-aquaagent odra_data.csv --target chlorophyll --mode autonomous --model autosklearn --impute autoencoder
```

---

## 📂 Structure

```
aquaagent/
├── agent.py
├── modules/
│   ├── analyzer.py
│   ├── data_handler.py
│   ├── imputation_engine.py
│   ├── ml_selector.py
│   ├── auto_optimizer.py
│   ├── visualizer.py
│   ├── reporter.py
│   └── model_explainer.py
└── templates/
    ├── classification_template.html
    └── suggestion_template.html
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to fork the repo and submit pull requests.

## 📜 License

MIT License © Tymoteusz Miller, 2025
