from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(models, X_test, y_test):
    for model_info in models:
        name=model_info['name']
        model=model_info['model']

        y_pred=model.predict(X_test)
        accuracy=accuracy_score(y_test, y_pred)
        
        print(f'Model: {name}')
        print(f'Accuracy: {accuracy:.4f}')
        print(classification_report(y_test, y_pred))
        print('-' * 50)