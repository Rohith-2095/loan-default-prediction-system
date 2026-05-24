import joblib



def save_model(model, path):

    joblib.dump(model, path)



def load_model(path):

    model = joblib.load(path)

    return model 