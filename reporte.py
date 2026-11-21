import mlflow
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("sqlite:///mlflow.db")
exp = mlflow.get_experiment_by_name("Proyecto_Final_Grado")
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
runs = runs[(runs["status"] == "FINISHED") & runs["metrics.cv_accuracy_mean"].notna()]
runs = runs.sort_values("start_time").drop_duplicates(subset="tags.mlflow.runName", keep="last")

tabla = runs[[
    "tags.mlflow.runName", "params.n_estimators", "params.max_depth",
    "params.min_samples_leaf", "metrics.train_accuracy", "metrics.test_accuracy",
    "metrics.cv_accuracy_mean", "metrics.cv_accuracy_std",
]].copy()
tabla.columns = ["Run", "n_estimators", "max_depth", "min_samples_leaf",
                 "Train acc", "Test acc", "CV mean", "CV std"]
tabla["Gap (train-CV)"] = tabla["Train acc"] - tabla["CV mean"]
tabla = tabla.sort_values("Run").round(4)
tabla.to_csv("tabla_comparativa_final.csv", index=False, sep=";", encoding="utf-8-sig")
tabla.to_excel("tabla_comparativa_final.xlsx", index=False)
print(tabla.to_string(index=False))

ax = tabla.set_index("Run")[["Train acc", "Test acc", "CV mean"]].plot(
    kind="bar", figsize=(11, 5), ylim=(0.8, 1.02))
ax.set_title("Comparación de experimentos - Random Forest (Iris)")
ax.set_ylabel("Accuracy")
ax.legend(loc="lower right")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("grafico_metricas_final.png", dpi=150)