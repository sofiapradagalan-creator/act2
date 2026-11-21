import mlflow, mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

EXPERIMENT = "Proyecto_Final_Grado"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT)

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

configs = [
    {"name": "baseline_comparable", "n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
    {"name": "Run_1_n50_depth3",   "n_estimators": 50,  "max_depth": 3,  "min_samples_leaf": 1},
    {"name": "Run_2_n150_depth5",  "n_estimators": 150, "max_depth": 5,  "min_samples_leaf": 1},
    {"name": "Run_3_n200_depth10", "n_estimators": 200, "max_depth": 10, "min_samples_leaf": 1},
    {"name": "run_02_pocos_arboles_poco_profundos", "n_estimators": 10,  "max_depth": 2,    "min_samples_leaf": 1},
    {"name": "run_03_depth3",                       "n_estimators": 50,  "max_depth": 3,    "min_samples_leaf": 1},
    {"name": "run_04_muchos_arboles_sin_limite",    "n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
    {"name": "run_05_regularizado",                 "n_estimators": 300, "max_depth": 5,    "min_samples_leaf": 3},
]

for cfg in configs:
    params = {k: v for k, v in cfg.items() if k != "name"}
    with mlflow.start_run(run_name=cfg["name"]):
        mlflow.log_params(params)

        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        cv = cross_val_score(RandomForestClassifier(**params, random_state=42),
                             X_train, y_train, cv=5)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("cv_accuracy_mean", cv.mean())
        mlflow.log_metric("cv_accuracy_std", cv.std())
        mlflow.log_metric("overfit_gap", train_acc - test_acc)
        mlflow.sklearn.log_model(model, name="iris_model",
                                 skops_trusted_types=["sklearn.tree._tree.Tree"])
        print(f"{cfg['name']}: train={train_acc:.3f} test={test_acc:.3f} "
              f"cv={cv.mean():.3f}±{cv.std():.3f}")

# ---------- Tabla comparativa y gráfico ----------
exp = mlflow.get_experiment_by_name(EXPERIMENT)
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
runs = runs[runs["tags.mlflow.runName"].isin([c["name"] for c in configs])]

tabla = runs[[
    "tags.mlflow.runName", "params.n_estimators", "params.max_depth",
    "params.min_samples_leaf", "metrics.train_accuracy", "metrics.test_accuracy",
    "metrics.cv_accuracy_mean", "metrics.cv_accuracy_std", "metrics.overfit_gap",
]].copy()
tabla.columns = ["Run", "n_estimators", "max_depth", "min_samples_leaf",
                 "Train acc", "Test acc", "CV mean", "CV std", "Gap (train-test)"]
tabla = tabla.sort_values("Run").round(4)
tabla.to_csv("tabla_comparativa.csv", index=False)
print("\n", tabla.to_string(index=False))

ax = tabla.set_index("Run")[["Train acc", "Test acc", "CV mean"]].plot(
    kind="bar", figsize=(11, 5), ylim=(0.8, 1.02))
ax.set_title("Comparación de experimentos - Random Forest (Iris)")
ax.set_ylabel("Accuracy")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("grafico_metricas.png", dpi=150)