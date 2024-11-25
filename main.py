import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics.pairwise import cosine_similarity

file_path = os.path.join(os.path.dirname(__file__), 'data.csv')

try:
    data = pd.read_csv(file_path, delimiter=',')
    print("Failas sėkmingai įkeltas!")
except Exception as e:
    print(f"Klaida įkeliant failą: {e}")
    exit()

X = data[['Lytis', 'Amzius(men)', 'Atvykimo ketvirtis', 'Spalva', 'Dydis']]
y = data['Trukme(dienos)']

y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

categorical_features = ['Lytis', 'Spalva']
numeric_features = ['Amzius(men)', 'Atvykimo ketvirtis', 'Dydis']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(random_state=42, objective='reg:squarederror'))
])

param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [3, 6, 10],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    model_pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Geriausi parametrai: {grid_search.best_params_}")

y_pred_log = best_model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Grįžtame iš log-transformacijos
mae = mean_absolute_error(np.expm1(y_test), y_pred)  # Atstatome tikslinio kintamojo reikšmes
r2 = r2_score(np.expm1(y_test), y_pred)

print(f"Vidutinė absoliuti klaida (MAE): {mae}")
print(f"R2 reikšmė: {r2}")

print("\n--- Naujo šuns duomenų įvedimas ---")
lytis = input("Įveskite šuns lytį (m/v): ").strip()
amzius_men = int(input("Įveskite šuns amžių mėnesiais: ").strip())
atvykimo_ketvirtis = int(input("Įveskite atvykimo ketvirtį (1-4): ").strip())
spalva = input("Įveskite šuns spalvą: ").strip()
dydis = int(input("Įveskite šuns dydį (1 - mažas, 2 - vidutinis, 3 - didelis): ").strip())

naujas_suo = pd.DataFrame({
    'Lytis': [lytis],
    'Amzius(men)': [amzius_men],
    'Atvykimo ketvirtis': [atvykimo_ketvirtis],
    'Spalva': [spalva],
    'Dydis': [dydis]
})

try:
    prognoze_log = best_model.predict(naujas_suo)
    prognoze = np.expm1(prognoze_log)
    print(f"\nPrognozuojama, kiek laiko šuo praleis prieglaudoje: {prognoze[0]:.2f} dienų")
except Exception as e:
    print(f"Įvesti duomenys netinkami prognozei: {e}")

# Rasti šunį su panašiais duomenimis pagal amžių, dydį ir spalvą
similar_dog = data[
    (data['Amzius(men)'] == amzius_men) & 
    (data['Dydis'] == dydis) & 
    (data['Spalva'] == spalva)
]

if not similar_dog.empty:
    similar_time = similar_dog['Trukme(dienos)'].mean()
    print(f"Vidutinė praleista laiką panašaus šuns prieglaudoje: {similar_time:.2f} dienos")

    # Parodyti diagramą
    plt.figure(figsize=(8, 5))
    plt.bar(['Prognozuotas laikas', 'Panašaus šuns laikas'], 
            [prognoze[0], similar_time], color=['blue', 'lightblue'])
    plt.title('Palyginimas tarp prognozuoto laiko ir panašaus šuns praleisto laiko')
    plt.ylabel('Laikas praleistas prieglaudoje (dienos)')
    plt.show()

else:
    print("Panašaus šuns su tokiais duomenimis nerasta.")
