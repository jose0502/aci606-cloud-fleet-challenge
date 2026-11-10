"""Exercise 4 training starter - Implemented."""
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

DATA = Path("data/training.csv")
MODEL_PATH = Path("models/model.joblib")

FEATURES = [
    "speed",
    "battery",
    "motor_temp",
    "vibration",
    "motor_current",
    "obstacle_distance",
]
TARGET = "status"

def main() -> None:
    if not DATA.exists():
        raise SystemExit(
            "Training dataset not found. Run: python simulator/generate_dataset.py"
        )

    df = pd.read_csv(DATA)
    print(df.head())
    print("\nClass distribution:")
    print(df[TARGET].value_counts())

    X = df[FEATURES]
    y = df[TARGET]

    # 1. Separación train/test
    if 'fleet_id' in df.columns:
        print("\nSeparando datos por fleet_id (GroupShuffleSplit)...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=df['fleet_id']))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        print("\nAdvertencia: 'fleet_id' no encontrado. Usando separación estratificada...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

    # 2. Definir clasificadores
    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])

    pipe_rf = Pipeline([
        ('scaler', StandardScaler()), 
        ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])

    models = {
        'Logistic Regression': pipe_lr, 
        'Random Forest': pipe_rf
    }

    best_model_name = None
    best_f1 = -1
    best_model = None

    # 3. Entrenar y evaluar
    for name, model in models.items():
        print(f"\n--- Entrenando {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"F1 Macro: {f1:.4f}")
        
        print("Matriz de Confusión:")
        print(confusion_matrix(y_test, y_pred))
        
        print("Reporte de Clasificación:")
        print(classification_report(y_test, y_pred))

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print(f"\n Modelo seleccionado: {best_model_name} (F1 Macro: {best_f1:.4f})")
    
    # 4. Guardar el modelo
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nModelo guardado exitosamente en: {MODEL_PATH}")

    # --- 5. GENERAR GRÁFICOS DEL EDA ---
    print("\nGenerando gráficos del EDA...")
    Path("data").mkdir(exist_ok=True)

    # Gráfico 1: Distribución de Clases
    plt.figure(figsize=(6, 4))
    colores = {'NORMAL': '#2ea043', 'WARNING': '#d29922', 'CRITICAL': '#f85149'}
    sns.countplot(x=y, palette=colores)
    plt.title('Distribución de Clases (EDA)')
    plt.xlabel('Estado')
    plt.ylabel('Cantidad')
    plt.tight_layout()
    plt.savefig('data/class_distribution.png')
    plt.close()
    
    # Gráfico 2: Matriz de Confusión
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.title(f'Matriz de Confusión - {best_model_name}')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig('data/confusion_matrix.png')
    plt.close()

    # Gráfico 3: Matriz de Correlación (NUEVO)
    # Para correlacionar el status, lo pasamos a números: NORMAL=0, WARNING=1, CRITICAL=2
    df_corr = df[FEATURES + [TARGET]].copy()
    df_corr['status_num'] = df_corr[TARGET].map({'NORMAL': 0, 'WARNING': 1, 'CRITICAL': 2})
    
    # Calculamos la correlación de Pearson
    corr_matrix = df_corr[FEATURES + ['status_num']].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Matriz de Correlación de Variables vs Estado')
    plt.tight_layout()
    plt.savefig('data/correlation_matrix.png')
    plt.close()

    print(" Gráficos guardados exitosamente en la carpeta data/")


if __name__ == "__main__":
    main()