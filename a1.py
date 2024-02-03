import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score


class TextProcessor:
    def __init__(self, configurations):
        self.configurations = configurations

    def preprocess_text(self, text, lemmatize, rm_stopwords, rm_char):
        text = text.lower()
        text = ''.join([char for char in text if char.isalpha() or char == ' ']) if rm_char else text
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in stopwords.words('english')] if rm_stopwords else tokens
        tokens = [WordNetLemmatizer().lemmatize(token) if lemmatize else token for token in
                  tokens]
        return ' '.join(tokens)

    def preprocess_datasets(self, df):
        datasets = {}
        for config in self.configurations:
            config_str = f"df_lemmatize_{config['lemmatize']}_rmstopwords_{config['rm_stopwords']}_rmchar_{config['rm_char']}"
            df[config_str] = df['text'].apply(self.preprocess_text, **config)
            datasets[config_str] = df[[config_str, 'class']]
        return datasets


def load_data():
    facts = np.loadtxt("facts.txt", delimiter='\t', dtype='str')
    fakes = np.loadtxt("fakes.txt", delimiter='\t', dtype='str')
    df = pd.DataFrame([(fact, 'fact') for fact in facts] + [(fake, 'fake') for fake in fakes],
                      columns=['text', 'class'])
    return df


def train_models(datasets):
    models = {
        'Perceptron': {
            'model': Perceptron(),
            'params': {'alpha': [0.0001, 0.001, 0.01, 0.1], 'penalty': ['l2', 'l1']}
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=123),
            'params': {'C': [0.1, 1, 2, 5, 10, 15], 'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear']}
        },
        'LinearSVC': {
            'model': LinearSVC(class_weight='balanced', dual='auto'),
            'params': {'C': [0.1, 1, 10]}
        }
    }

    results = []

    for config_str, dataset in datasets.items():
        print(f"Training Model for dataset: {config_str}")  # Output the current dataset being processed
        X_train, X_test, y_train, y_test = train_test_split(dataset[config_str], dataset['class'], test_size=20,
                                                            random_state=42, stratify=dataset['class'])

        vectorizers = {'TF-IDF': TfidfVectorizer(), 'Count': CountVectorizer()}
        for vect_name, vectorizer in vectorizers.items():
            X_train_vect = vectorizer.fit_transform(X_train)
            X_test_vect = vectorizer.transform(X_test)

            for model_name, model_info in models.items():
                grid = GridSearchCV(model_info['model'], model_info['params'], cv=5)
                grid.fit(X_train_vect, y_train)

                best_model = grid.best_estimator_
                y_pred_train = best_model.predict(X_train_vect)
                y_pred_test = best_model.predict(X_test_vect)

                results.append({
                    'config': config_str,
                    'vectorizer': vect_name,
                    'model': model_name,
                    'best_params': grid.best_params_,
                    'train_accuracy': accuracy_score(y_train, y_pred_train),
                    'test_accuracy': accuracy_score(y_test, y_pred_test),
                    'cross_val_train_score': cross_val_score(best_model, X_train_vect, y_train, cv=5).mean(),
                    'cross_val_test_score': cross_val_score(best_model, X_test_vect, y_test, cv=5).mean()
                })
    print(pd.DataFrame(results))
    pd.DataFrame(results).sort_values(by=['cross_val_test_score'], ascending=[False]).to_csv("nlp_results_3.csv")
    return pd.DataFrame(results)

def main():
    configurations = [
        {"lemmatize": True, "rm_stopwords": True, "rm_char": True},
        {"lemmatize": False, "rm_stopwords": True, "rm_char": True},
        {"lemmatize": True, "rm_stopwords": False, "rm_char": True},
        {"lemmatize": True, "rm_stopwords": True, "rm_char": False},
        {"lemmatize": True, "rm_stopwords": True, "rm_char": False},
        {"lemmatize": False, "rm_stopwords": True, "rm_char": False},
        {"lemmatize": True, "rm_stopwords": False, "rm_char": False},
        {"lemmatize": False, "rm_stopwords": False, "rm_char": False},
    ]

    df = load_data()
    processor = TextProcessor(configurations)
    datasets = processor.preprocess_datasets(df)
    results_df = train_models(datasets)

    # Sorting results_df
    sorted_results = results_df.sort_values(by=['test_accuracy', 'cross_val_test_score'], ascending=[False, False])

    # Return TOP SETTINGS
    top_result = sorted_results.iloc[0]
    print(f"Top Configuration and Model:")
    print(f"Configuration: {top_result['config']}")
    print(f"Vectorizer: {top_result['vectorizer']}")
    print(f"Model: {top_result['model']}")
    print(f"Training Accuracy: {top_result['train_accuracy']}")
    print(f"Testing Accuracy: {top_result['test_accuracy']}")
    print(f"Cross-Validation Train Score: {top_result['cross_val_train_score']}")
    print(f"Cross-Validation Test Score: {top_result['cross_val_test_score']}")


    return top_result.to_dict()


if __name__ == "__main__":
    top_result = main()
    print(top_result)  