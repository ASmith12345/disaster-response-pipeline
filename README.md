# disaster-response-pipeline

Summary
This is part of Udacity DataScience NanoDegree Data Engineering Project. The motivartion is to analyze social media messages during a disaster to classify relevance to particular supports.

File Structure
app/templates
-go.html - html file which receives user input and displays results
-master.html - html file which diplays visualisations
app/run.py - runs flask app
data
-caegories.csv: disaster categories
-messages.csv: disaster messages
-process_data.py - ETL pipeline which cleans and loads into a database

models
-train_classifier.py: ML pipeline which trains and evaluates models, then svaes in a pickle file

How to run
Run in the following order

Run the ETL pipeline from the root directory
python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db

Run the ML pipeline from the root directory
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

run the web app from the app directory by running the following commands
cd app
python run.py

in the web browser, go to http://127.0.0.1:3000/