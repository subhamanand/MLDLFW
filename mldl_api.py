import json

from flask import Flask, Response, jsonify, request, session
from flask_cors import CORS, cross_origin
import os
import boto3
import pandas as pd
import ml_algorithm as algo
import ml_algorithm_1 as algo1
import ml_selection as sel
import ml_selection_1 as sel1
import predictions
import predictions_1
from sklearn.externals import joblib
import csv

app = Flask(__name__) 
CORS(app) 
#api = Api(app, version='1.0', title='APIs for Python Functions', validate=False) ns = api.namespace('primality', 'Returns a list of all primes below a given upper bound')

#@app.route('/')
#def home():
#	return render_template('login.html')
@app.route("/api/ml/getHeaders", methods = ['POST'])
def getHeaders():
	fileName = request.get_json()['fileName']
	print('bucket name ****** ',fileName)

	response = {}
	dataset = pd.read_csv(fileName)
	print(list(dataset))
	# return response_string
	return jsonify(status=1,results= list(dataset))

@app.route('/train', methods=['POST'])
def train():
	content = request.get_json();
	fileName = content['fileName']
	target = content['target']
	algo_type = content['type']
	#preference = content['preference']
	#dateformat = content['dateformat']
	#datecolumn = content['datecolumn']
	#s3 = boto3.client('s3',aws_access_key_id='yyyyyyyy',aws_secret_access_key='xxxxxxxxxxx')
	#Call S3 to list current buckets
	#response = s3.list_buckets()
	#for bucket in response['Buckets']:
	#	print bucket['Name']
	data = pd.read_csv("./trainingFiles/"+fileName)
	if algo_type == 'restaurant sales':
		burgercol,burgermodel,milkshakemodel,mape = sel1.restaurant_ml(data,target)
		rest_json = '{"statusCode":"200","burgerModel":'+json.dumps(burgermodel)+',"milkShakeModel":'+json.dumps(milkshakemodel)+',"trainingAccuracy":"'+str(mape)+'","inputVar":'+json.dumps(burgercol)+',"usefulVar":'+json.dumps(burgercol)+'}'
		return rest_json
	if algo_type=='regression':
		model,useful_var,input_var,mape = sel.regression_selection(data,target)
		#useful_var,input_var,model,mape = algo.regression(data,target,model)
		#session['model'] = model
		session['useful'] = useful_var
		session['input'] = input_var
		json_useful_var = json.dumps(useful_var)
		json_input_var = json.dumps(input_var)

		out_json = '{"statusCode":"200","modelName":"'+model+'","trainingAccuracy":"'+str(mape)+'","usefulVar":'+json_useful_var+',"inputVar":'+json_input_var+'}'
		#joblib.dump(model,'model.pkl')
		return Response(out_json,status=200)
	else:
		model,useful_var,input_var,mape = sel.classification_selection(data,target)
		#useful_var,input_var,model,mape = algo.classification(data,target,model)
		#session['model'] = model
		session['useful'] = useful_var
		session['input'] = input_var

		json_useful_var = json.dumps(useful_var)
		json_input_var = json.dumps(input_var)
		out_json = '{"statusCode":"200","modelName":"'+model+'","trainingAccuracy":"'+str(mape)+'","usefulVar":"'+json_useful_var+'","inputVar":"'+json_input_var+'"}'
		#joblib.dump(model,'model.pkl')
		return Response(out_json,status=200)

@app.route('/predict', methods=['POST'])
def predict():
	content = request.get_json()
	#model = session['model']
	# useful_var = session['useful']
	# input_var = session['input']
	#burgerModel = content['burgerModel']
	#milkShakeModel = content['milkShakeModel']
	#burgerColumns = content['burgerColumns']
	#dateRange = content['dateRange']
	useful_var = content['usefulVar']
	input_var = content['inputVar']
	fileName = content['fileName']
	modelName = content['modelName']
	model = joblib.load('./model/'+modelName)
	data = pd.read_csv("./input/"+fileName)
	useful_var=useful_var.split(',')
	input_var=input_var.split(',')
	#if burgerModel != None:
	file_path = predictions.prediction(data,model,useful_var,input_var,fileName)
	out_json = '{"statusCode":"200","outputFile":"'+file_path+'"}'
	return Response(out_json,status=200)

@app.route('/restaurant_predict', methods=['POST'])
def restaurant_predict():
	content = request.get_json()
	#model = session['model']
	# useful_var = session['useful']
	# input_var = session['input']
	m1 = content['models'][0]
	m2 = content['models'][1]
	#burgerColumns = content['burgerColumns']
	dateRange = content['dateRange']
	#useful_var = content['usefulVar']
	input_var = content['inputVar']
	#fileName = content['fileName']
	#modelName = content['modelName']
	burgerModel = joblib.load('./model/'+m1)
	milkShakeModel = joblib.load('./model/'+m2)
	#data = pd.read_csv("./input/"+fileName)
	#useful_var=useful_var.split(',')
	input_var=input_var.split(',')
	file_path = predictions_1.prediction_restaurant(dateRange,input_var,burgerModel,milkShakeModel)
	out_json = '{"statusCode":"200","outputFile":"'+file_path+'"}'
	return Response(out_json,status=200)

@app.route("/api/ml/getCsvData", methods = ['POST'])
def getCsvData():
	fileName = request.get_json()['fileName']
	print('bucket name ****** ',fileName)
	try:
		dataset = pd.read_csv(fileName)
		print(list(dataset))
		list_of_cols = list(dataset)

		data = {}
		data['columns'] = []

		for item in list_of_cols:
			data['columns'].append({
				'title': str(item),
				'mData': str(item)
			})

		print(data['columns'])

		with open(fileName, 'r') as csvfile:
			json_output = []
			reader = csv.DictReader(csvfile)
			for row in reader:
				# print(dict(row))
				json_output.append(dict(row))

			print(json_output)
		# with open('data_headers.json', 'w') as outfile:
		# 	json.dump(data, outfile, indent=4, sort_keys=True)
		return jsonify({'status':1,'columns':data,'data':json_output})
	except:
		return jsonify({'status':0})

if __name__ == "__main__":
	app.secret_key="abc"
	app.debug=True
	app.run()
