from flask import Flask, render_template, request
import pickle

# Load the pre-trained machine learning model
with open("model2.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the 24 input values from the form
    # for i in range(24):
    #     input_data = [float(request.form[f'input{{ i+1 }}'])]
    
    age=float(request.form['age'])
    bp=float(request.form['bp'])
    sg=float(request.form['sg'])
    al=float(request.form['al'])
    su=float(request.form['su'])
    rbc=float(request.form['rbc'])
    pc=float(request.form['pc'])
    pcc=float(request.form['pcc'])
    ba=float(request.form['ba'])
    bgr=float(request.form['bgr'])
    bu=float(request.form['bu'])
    sc=float(request.form['sc'])
    sod=float(request.form['sod'])
    pot=float(request.form['pot'])
    hemo=float(request.form['hemo'])
    pcv=float(request.form['pcv'])
    wc=float(request.form['wc'])
    rc=float(request.form['rc'])
    
    #Make a prediction using the pre-trained model
    input_d = [age,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc]
    prediction = model.predict([input_d])[0]
    if prediction==0:
        pp= "Safe zone, check our health plans"
    else:
        pp= "You're likely to have CKD, have a check on our page for more info"

    # Render the result template with the prediction value
    # if prediction==1:
    #     return "Chance to have ckd"
    # else:
    #     return "no ckd"
    return render_template('index.html', prediction=pp)

if __name__ == '__main__':
    app.run(debug=True, port=8000)