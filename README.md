# 🏭 MLOps Monitoring Dashboard - Maintenance Prédictive Industrielle

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Description

Projet académique de monitoring MLOps appliquée à la **maintenance prédictive industrielle**. 
Système de surveillance continue pour modèle de classification sur données IoT, avec détection automatisée de *data drift* capteurs.

### Contexte
- **Cycle Ingénieur - MLOps**
- **Date :** Avril 2026
- **Objectif :** Implémenter un pipeline complet de monitoring de modèle ML

## 🎯 Fonctionnalités

- ✅ Entraînement et sauvegarde d'un modèle Random Forest
- ✅ Simulation de data drift progressive
- ✅ Calcul périodique des métriques (Accuracy, Precision, Recall, F1-Score)
- ✅ Tableau de bord interactif Streamlit avec visualisations Plotly
- ✅ Détection automatisée de dérive avec alertes
- ✅ Historique des performances stocké dans un CSV

## 📁 Structure du Projet
```ml_monitoring_dashboard/
├── data/         # Données de métriques historiques
├── models/       # Artefacts (modèle, scaler, features)
├── src/          # Code source
│ ├── train_model.py        # Entraînement du modèle
│ ├── init_metrics.py       # Initialisation du CSV
│ ├── periodic_metrics.py   # Calcul périodique des métriques
│ └── dashboard.py          # Application Streamlit
├── requirements.txt     # Dépendances Python
├── .gitignore
└── README.md            # Documentation
```

## 🚀 Installation

### Prérequis
- Python 3.9 ou supérieur
- Git

### Étapes

1. **Cloner le repository**
```
git clone https://github.com/VOTRE_USERNAME/ml_monitoring_dashboard.git
cd ml_monitoring_dashboard ```

```

2. **Créer un environnement virtuel**
#### Windows
```
python -m venv venv
```
```
venv\Scripts\activate
```

#### Linux/Mac
```
python3 -m venv venv
```
```
source venv/bin/activate 
```

3. **Installer les dépendances**
```
pip install -r requirements.txt
```

## 📖 Utilisation

1. **Entraîner le modèle**
```
python src/train_model.py
```

2. **Initialiser le fichier de métriques**
``` 
python src/init_metrics.py
```

3. **Lancer la simulation de dérive**
```
python src/periodic_metrics.py
```

4. **Visualiser les résultats**
```
streamlit run src/dashboard.py
```

Le tableau de bord sera accessible sur http://localhost:8501
## 📊 Résultats Attendus
- Batches 1-5 : Performance stable (Accuracy ≈ 0.97-0.99)
- Batches 6-20 : Dégradation progressive due au data drift
- Alertes : Déclenchement automatique lorsque l'accuracy baisse de >5% ou >10%

## 📈 Métriques Suivies
- Accuracy: Exactitude globale
- Precision: Précision des prédictions positives
- Recall: Rappel (taux de détection)
- F1-Score: Moyenne harmonique

## 🎓 Contexte Académique
Ce projet s'inscrit dans le cadre du cours MLOps du Cycle Ingénieur. Il démontre la maîtrise des concepts suivants :
- Data Drift et Concept Drift
- Monitoring continu de modèles ML
- Visualisation de données temps réel
- Bonnes pratiques MLOps (versioning, séparation code/données)

## 📚 Ressources
- Documentation Streamlit
- Scikit-learn
 -Plotly Python

## 📄 License
Ce projet est distribué sous licence MIT - voir le fichier LICENSE pour plus de détails.
```
- 👨‍💻 Auteur : Ba Mohamadou Adama  |  Ingénieur IA - Spécialité Machince Learning & Data Science
- Email : medcapt66@gmail.com
