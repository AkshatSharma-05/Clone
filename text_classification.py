import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def main_method(new_emails):

    df = pd.read_csv('./data.csv')

    accuracy = 0
    predictions = []

    # Correct columns
    X = df['subject'] + " " + df['body']
    y = df['topic']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Vectorization
    vectorizer = CountVectorizer()
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # vectorizer = TfidfVectorizer()

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)

    # Test predictions
    y_pred = classifier.predict(X_test_vec)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Predict new emails
    new_emails_vec = vectorizer.transform(new_emails)
    predictions = classifier.predict(new_emails_vec)

    return predictions, accuracy

# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(max_iter=1000)

# from sklearn.svm import LinearSVC
# classifier = LinearSVC()

# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators=100)

# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors=5)

# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier()
