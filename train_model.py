import os
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def main():
    path = os.path.join('dataset', 'Parkinsson disease.csv')
    df = pd.read_csv(path)

    # prepare features and target
    df = df.drop(columns=['name'], errors='ignore')
    X = df.drop(columns=['status'])
    y = df['status']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

    pipe = make_pipeline(MinMaxScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Test accuracy: {acc:.4f}')

    os.makedirs('models', exist_ok=True)
    joblib.dump(pipe, os.path.join('models', 'model.pkl'))
    print('Saved model to models/model.pkl')


if __name__ == '__main__':
    main()
