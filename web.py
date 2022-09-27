from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model2=pickle.load(open('model2.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
    print(request.values)

    inputs=[float(i) for i in request.values.values()]
    array_features = [np.array(inputs)]
    result=model2.predict(array_features)
    result=result.item()
    result=round(result)
    if result==0:
        output='setosa'
    elif result == 1:
        output='versicolor'
    else:
        output='verginica'


    return render_template ('result.html',prediction_text="Predicted iris flower category is {} ".format(output))

if __name__=='__main__':
    app.run(port=8000)


