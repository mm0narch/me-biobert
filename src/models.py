from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tqdm import tqdm

models=[{
    'name':'SVC',
    'model':SVC(
        kernel='linear',
        probability=True,
        random_state=42
    )
},{
    'name': 'Decision Tree',
    'model': DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        criterion='gini',
        splitter='best',
        random_state=42
    )
},{
    'name': 'Random Forest',
    'model': RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
},{
    'name': 'XGBoost',
    'model': XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        eval_metric='mlogloss'
        )
}]

def train_model(models, X_train, y_train):
    for model_info in tqdm(models, desc='Training models', unit='model'):
        name=model_info['name']
        model=model_info['model']
        model.fit(X_train, y_train)
    
    return models