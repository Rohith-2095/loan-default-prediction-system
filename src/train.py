from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


models = {
    'Logistic Regression': LogisticRegression(),

    'Decision Tree': DecisionTreeClassifier(),

    'Random Forest': RandomForestClassifier(),

    'XGBoost': XGBClassifier(),

    'SVM': SVC(probability=True),

    'KNN': KNeighborsClassifier(),

    'Naive Bayes': GaussianNB()
}



def train_models(X_train, y_train):

    trained_models = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models