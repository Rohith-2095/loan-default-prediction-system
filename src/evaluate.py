from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier



def evaluate_models(models, X_test, y_test):

    for name, model in models.items():

        y_pred = model.predict(X_test)

        print(f'\n{name}')

        print('Accuracy:', accuracy_score(y_test, y_pred))

        print(classification_report(y_test, y_pred))

        print(confusion_matrix(y_test, y_pred))



def cross_validation(model, X, y):

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5
    )

    print('Cross Validation Mean:', scores.mean())



def hyperparameter_tuning(X_train, y_train):

    params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10]
    }

    grid = GridSearchCV(
        RandomForestClassifier(),
        params,
        cv=5
    )

    grid.fit(X_train, y_train)

    print('Best Parameters:', grid.best_params_)

    return grid.best_estimator_