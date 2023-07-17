from os import name
from flask import Flask, render_template, request
from flask_mysqldb import MySQL
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from flask import send_file
import io
from sklearn.tree import export_graphviz
import pydotplus


application = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


test = pd.read_csv("test_data.csv")
train = pd.read_csv("training_data.csv")

x_test = test.drop('prognosis', axis=1)
y_test = test.prognosis


@application.route('/')
def Landing_page():
    return render_template('index.html')


@application.route('/check')
def check_page():
    return render_template('letscheck.html')


@application.route('/about')
def about_page():
    return render_template('about.html')


@application.route('/team')
def team_page():
    return render_template('team.html')


@application.route('/contact')
def contact_page():
    return render_template('contact.html')

    if request.method == 'POST':
        name = request.form['name']
        address = request.form['address']
        email = request.form['email']
        phone = request.form['phone']
        message = request.form['message']
        cur = mysql.connection.cursor()
        cur.execute(" INSERT INTO contacts(name, address, email, phone, message)  VALUES( %s, %s, %s, %s, %s )",
                    (name, address, email, phone, message))
        mysql.connection.commit()
        cur.close()
        msg = Message(
                'Medicine ',
                sender='medicination1@gmail.com',
                recipients=[email]
               )

        msg.html = render_template('mail.html')
        mail.send(msg)

        return render_template('ThankYou.html')

    return render_template('contact.html')


@application.route('/predict', methods=['POST', 'GET'])

def predict():
    if request.method == 'POST':
        col = x_test.columns
        inputt = [str(x) for x in request.form.values()]

        # Check if all the input symptoms are among the 132 symptoms in the dataset
        if not all(symptom in col for symptom in inputt):
            return render_template('letscheck.html', pred="No symptoms found")

        b = [0]*132
        for x in range(0, 132):
            for y in inputt:
                if col[x] == y:
                    b[x] = 1
        b = np.array(b)
        b = b.reshape(1, 132)
        prediction = model.predict(b)
        prediction = prediction[0]
        print(prediction)

        # Calculate the accuracy score
        accuracy = accuracy_score(y_test, model.predict(x_test))

    return render_template('letscheck.html', symptoms=inputt, pred="The probable diagnosis says it could be {}".format(prediction), accuracy=f"The accuracy score is {accuracy}")

    application.run(debug=True, port=8001)

    from flask import Flask, render_template, request, send_file


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
test = pd.read_csv("test_data.csv")
x_test = test.drop('prognosis', axis=1)
y_test = test.prognosis
cols = x_test.columns


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
      inputt = list(request.form.values())
        inputt = [str(x) for x in inputt]
        if len(inputt) < 2:
            return render_template('index.html', pred="At least 2 symptoms are required")
        elif len(inputt) > 5:
            return render_template('index.html', pred="Maximum 5 symptoms are allowed")

        if not all(symptom in cols for symptom in inputt):
            return render_template('index.html', pred="No symptoms found")
        b = [0]*len(cols)
        for i, col in enumerate(cols):
            if col in inputt:
                b[i] = 1
        b = np.array(b).reshape(1, -1)
        pred = model.predict(b)[0]
        accuracy = accuracy_score(y_test, model.predict(x_test))
       dot_data = export_graphviz(model, out_file=None, feature_names=cols, class_names=model.classes_,
                           filled=True, rounded=True, special_characters=True)

# Render the tree as an image using Graphviz
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render('decision_tree')
        return render_template('index.html', symptoms=inputt, pred=pred, accuracy=accuracy)

@app.route('/decision-tree')
def decision_tree():
    return send_file("decision_tree.png", mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=8001)



import numpy as np

# Define a function to calculate the Gini index
def calc_gini(data, target):
    n = len(data)
    gini = 0
    for c in np.unique(target):
        p = np.sum(target == c) / n
        gini += p * (1 - p)
    return gini

# Define a function to calculate the Gini index for a split
def calc_split_gini(data, feature, target):
    n = len(data)
    left = data[feature] < data[feature].mean()
    right = ~left
    gini_left = calc_gini(data[left], target[left])
    gini_right = calc_gini(data[right], target[right])
    return gini_left * sum(left) / n + gini_right * sum(right) / n

# Define a function to find the best split for a given feature
def find_best_split(data, feature, target):
    split_gini = [calc_split_gini(data, feature, target) for feature in data.columns]
    best_feature = np.argmin(split_gini)
    return best_feature, split_gini[best_feature]

# Define the Node class to represent a decision tree node
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

# Define the DecisionTree class to build and use a decision tree
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stop criteria
        if depth == self.max_depth or n_labels == 1 or n_samples < 2:
            return Node(value=np.argmax(np.bincount(y)))

        # Find the best split
        best_feature, best_gini = find_best_split(X, y, n_features)
        if best_gini == 0:
            return Node(value=np.argmax(np.bincount(y)))

        # Recursive splitting
        left = X.iloc[:, best_feature] < X.iloc[:, best_feature].mean()
        right = ~left
        left_tree = self._build_tree(X.loc[left], y[left], depth+1)
        right_tree = self._build_tree(X.loc[right], y[right], depth+1)

        return Node(best_feature, X.iloc[:, best_feature].mean(), left_tree, right_tree)

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)
