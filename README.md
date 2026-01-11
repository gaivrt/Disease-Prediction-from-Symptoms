# Disease Prediction from Symptoms

A machine learning system for predicting diseases based on symptoms.

## Algorithms

- Naive Bayes
- Decision Tree
- Random Forest

## Dataset

Source: [Kaggle - Disease Prediction Using Machine Learning](https://www.kaggle.com/kaushil268/disease-prediction-using-machine-learning)

- 132 symptom features
- 41 disease classes
- 4920 training samples, 42 test samples

## Project Structure

```
src/
├── main.py           # CLI entry point
├── train.py          # Training module
├── evaluate.py       # Evaluation module
├── demo.py           # Gradio control panel
├── models/           # Model definitions
│   ├── naive_bayes.py
│   ├── decision_tree.py
│   └── random_forest.py
├── data/
│   └── loader.py
└── analysis/
    └── interaction.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
cd src

# Train all models
python main.py train

# Train specific model
python main.py train --model random_forest

# Evaluate all models
python main.py evaluate

# Evaluate specific model
python main.py evaluate --model naive_bayes
```

### Gradio Web Interface

```bash
cd src
python demo.py
```

Features:
- **Train Tab**: Train models and view validation metrics
- **Evaluate Tab**: Evaluate models on test set with Accuracy and Macro F1
- **Predict Tab**: Interactive prediction with optional verification

---

*Disclaimer: This project is for educational purposes only. Please consult a Doctor for actual medical advice.*
