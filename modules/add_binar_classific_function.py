def compare_models_binar_classific(
    df, target_col, test_size=0.2, random_state=42, n_components=15
):
    # Импорт необходимых библиотек
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from catboost import CatBoostClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    from umap import UMAP
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, roc_curve, auc
    )
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.patches as mpatches

    # Признаки и целевая переменная
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Перечень моделей 
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(random_state=random_state),
        'RandomForest': RandomForestClassifier(random_state=random_state),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=random_state),
        'MLPClassifier': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=random_state),
        'SVC': SVC(probability=True)
    }

    # Параметры для GridSearch
    param_grids = {
        'LogisticRegression': {'C': [0.1, 1, 10]},
        'DecisionTree': {'max_depth': [3, 5, 7, 10, None]},
        'RandomForest': {'n_estimators': [50, 100], 'max_depth': [5, 10, None]},
        'CatBoost': {'depth': [4, 6, 8], 'learning_rate': [0.01, 0.1]},
        'MLPClassifier': {'hidden_layer_sizes': [(100,), (100, 50)], 'max_iter': [500, 1000]},
        'SVC': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
    }

    reducers = {
        'Без снижения размерности': None,
        'PCA': PCA(n_components=n_components, random_state=random_state),
        'UMAP': UMAP(n_components=n_components, random_state=random_state)
    }

    X_dict, X_train_dict, X_test_dict = {}, {}, {}
    y_train_dict, y_test_dict = {}, {}

    for key, reducer in reducers.items():
        if reducer is None:
            X_red = X
        else:
            X_red = reducer.fit_transform(X)
        X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
            X_red, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_dict[key] = X_red
        X_train_dict[key] = X_train_red
        X_test_dict[key] = X_test_red
        y_train_dict[key] = y_train_red
        y_test_dict[key] = y_test_red

    # Обучение моделей с GridSearch и сбор метрик
    results_all = {key: {} for key in reducers}
    roc_curves_all = {key: {} for key in reducers}
    best_params_all = {key: {} for key in reducers}

    for key in reducers:
        for name, model in models.items():
            param_grid = param_grids[name]
            gs = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            gs.fit(X_train_dict[key], y_train_dict[key])
            best_model = gs.best_estimator_
            best_params_all[key][name] = gs.best_params_

            # Предсказания и вероятности
            y_pred = best_model.predict(X_test_dict[key])
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test_dict[key])[:, 1]
            else:
                y_proba = best_model.decision_function(X_test_dict[key])

            # Метрики
            results_all[key][name] = {
                'accuracy': accuracy_score(y_test_dict[key], y_pred),
                'precision': precision_score(y_test_dict[key], y_pred),
                'recall': recall_score(y_test_dict[key], y_pred),
                'f1': f1_score(y_test_dict[key], y_pred),
                'roc_auc': roc_auc_score(y_test_dict[key], y_proba)
            }

            # ROC-кривая
            fpr, tpr, _ = roc_curve(y_test_dict[key], y_proba)
            roc_auc_val = auc(fpr, tpr)
            roc_curves_all[key][name] = (fpr, tpr, roc_auc_val)

    # Вывод результатов
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    model_names = list(models.keys())
    mode_names = list(reducers.keys())

    for mode in mode_names:
        print(f"\n=== {mode} ===")
        for model in model_names:
            params = best_params_all[mode][model]
            res = results_all[mode][model]
            print(f"\n{model}:")
            print(f"  Лучшие параметры: {params}")
            print(f"  Accuracy: {res['accuracy']:.3f}")
            print(f"  Precision: {res['precision']:.3f}")
            print(f"  Recall: {res['recall']:.3f}")
            print(f"  F1: {res['f1']:.3f}")
            print(f"  ROC AUC: {res['roc_auc']:.3f}")

    # Визуализация метрик
    fig, axes = plt.subplots(len(metrics), 1, figsize=(23, 7*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    bar_height = 0.4

    for i, metric in enumerate(metrics):
        y_pos = np.arange(len(model_names))
        for j, mode in enumerate(mode_names):
            values = [results_all[mode][m][metric] for m in model_names]
            bars = axes[i].barh(
                y_pos + (j-1)*bar_height, values, height=bar_height, 
                color=['royalblue', 'orange', 'green'][j], 
                label=mode, alpha=0.85, edgecolor='black'
            )
            for idx, bar in enumerate(bars):
                axes[i].text(
                    bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{bar.get_width():.2f}",
                    va='center', ha='left', fontsize=17, color='black'
                )
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels([f"{name}" for name in model_names], fontsize=12)
        axes[i].set_xlabel(metric, fontsize=13)
        axes[i].set_title(f'Сравнение моделей по метрике: {metric}', fontsize=15, fontweight='bold')
        axes[i].xaxis.set_major_locator(mticker.MaxNLocator(5))
        axes[i].grid(axis='x', linestyle='--', alpha=0.5)
        if i == 0:
            handles = [mpatches.Patch(color=['royalblue', 'orange', 'green'][k], label=mode_names[k]) 
                      for k in range(len(mode_names))]
            axes[i].legend(handles=handles, loc='lower center', 
                         bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False, fontsize=12)

    plt.tight_layout()
    plt.show()

    # ROC-кривые
    plt.figure(figsize=(20, 6))
    for j, mode in enumerate(mode_names):
        plt.subplot(1, 3, j+1)
        for name in model_names:
            fpr, tpr, roc_auc_val = roc_curves_all[mode][name]
            plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_val:.2f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.title(f'ROC-кривые ({mode})', fontsize=14, fontweight='bold')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()