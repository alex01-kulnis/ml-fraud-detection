# Fraud Detection

Учебный проект бинарной классификации мошеннических транзакций с чистым разделением
генерации данных, feature engineering, оценки моделей, обучения и inference.

## ML flow

```text
synthetic raw data
→ validation and feature engineering
→ one stratified Train/Test split
→ Logistic Regression / Random Forest / CatBoost baseline CV
→ Random Forest and CatBoost tuning on Train
→ selection by mean CV ROC-AUC
→ OOF probabilities and F1 threshold tuning
→ final fit on full Train
→ one final Test evaluation
→ native model artifact + metadata.json
```

CatBoost CV, tuning и OOF выполняются вручную: это обходит несовместимость
`sklearn.clone()` с `CatBoostClassifier(cat_features=...)` в текущем Python 3.14
окружении. Preprocessing Logistic Regression и Random Forest находится внутри sklearn
Pipeline и обучается отдельно в каждом CV fold.

## Установка

```bash
python3.14 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Обучение

Полный запуск с настройками из `fraud_detection.config`:

```bash
.venv/bin/python scripts/train.py
```

Другие директория артефактов и размер synthetic dataset:

```bash
.venv/bin/python scripts/train.py --artifacts-dir artifacts --n-samples 3000
```

Результат:

```text
artifacts/
├── metadata.json
└── model.cbm | model.joblib
```

## Inference

Predictor принимает одну сырую транзакцию — derived features передавать не нужно:

```python
from fraud_detection.inference import FraudPredictor

predictor = FraudPredictor.load("artifacts")
prediction = predictor.predict(
    {
        "age": 34,
        "income": 4200.0,
        "amount": 950.0,
        "avg_amount_30d": 180.0,
        "transactions_24h": 8,
        "account_age_days": 45,
        "transaction_hour": 2,
        "city": None,
        "device": "Android",
        "is_new_device": 1,
    }
)
print(prediction)
# {'fraud_probability': 0.73, 'is_fraud': True, 'threshold': 0.35}
```

## Проверки

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
```

Smoke-тест использует уменьшенные модели и не запускает полный тяжёлый tuning.

## Структура

```text
src/fraud_detection/
├── config.py
├── data/dataset.py
├── features/engineering.py
├── preprocessing/builders.py
├── models/factories.py
├── evaluation/{cross_validation,metrics,tuning}.py
├── training/{artifacts,pipeline,train}.py
└── inference/predictor.py
```
