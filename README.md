# Rent Predictor

Simple rent prediction app using a linear regression model trained on the provided CSVs.

Quick start

```bash
python3 -m pip install -r requirements.txt
python train.py        # trains model and saves to ./model/
streamlit run app.py   # open UI in browser
```

Inputs: `BHK`, `Size` (sqft), `City` → output: predicted monthly rent.
