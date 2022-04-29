
from flask import Flask, request, render_template, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import preprocess_kgptalkie as ps
import joblib
import re



app = Flask(__name__)


#Process and Clean Data
def get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = ps.cont_exp(x)
    x = ps.remove_emails(x)
    x = ps.remove_urls(x)
    x = ps.remove_html_tags(x)
    x = ps.remove_rt(x)
    x = ps.remove_accented_chars(x)
    x = ps.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x


@app.route("/", methods =["GET", "POST"])
def index():
	return render_template("index.html")


@app.route("/process", methods =["POST"])
def process():
	comment = request.form['comment']
	print(comment)
	comment = get_clean(comment) #Clean data

	vectorizer = joblib.load('vectorizer.pkl') #Load in Vectorizer
	vec = vectorizer.transform([comment])  
	print(vec.shape)

	model = joblib.load('model.joblib') #Load the model
	pred = model.predict(vec) #Predict Sentiment
	print("Prediction: ", pred)

	pred_result = list(pred)[0]
	print("Prediction Result: ", pred_result)


	return jsonify({'response':pred_result})



#Run App
if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=80)