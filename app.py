from flask import Flask, render_template,request,redirect,url_for,jsonify
from flask_cors import CORS
import os
import pandas as pd
import joblib
import psycopg2 #importing psycopg2 to connect to postgresql database in supabase
app=Flask(__name__)  #Flask app instance creation
CORS(app) #enabling CORS for flask app
model=joblib.load("habitability_trained.pkl") #loading the trained model:)

def get_db_connection():
    conn=psycopg2.connect( #connecting to postgresql database in supabase
        host=os.getenv("SUPABASE_DB_HOST"),
        user=os.getenv("SUPABASE_DB_USER"),
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        port=os.getenv("SUPABASE_DB_PORT"),
        database=os.getenv("SUPABASE_DB_DATABASE")
    )
    return conn

@app.route('/',methods=["GET"])
def home():
    return jsonify({
        "status":"success",
        "message":"Welcome to ExoPlanet Habitability AI "
    })
    
@app.route("/db_test",methods=["GET"])
def db_test():
    try:
        conn=get_db_connection()
        cursor=conn.cursor()
        cursor.execute("SELECT 1;")
        result=cursor.fetchone()
        cursor.close()
        conn.close()
        return jsonify({
            "status":"success",
            "message":"Database connection successful",
            "result":result
        })
    except Exception as e:
        return jsonify({
            "status":"error",
            "message":"Database connection failed",
            "error":str(e)
        }),500
        
@app.route('/predict',methods=["POST"])
def predict():
    data=request.get_json()# getting data from request by user in json format
    required_fields=[
        'radius',
        'mass',
        'temp',
        'orbital_period',
        'distance_star',
        'star_temp',
        'eccentricity',
        'semi_major_axis',
        'star_type'
    ]
    
    missing=[field for field in required_fields if field not in data]
    if missing:
        return jsonify({
            "error":"Missing fields",
            "missing_fields":missing
        }),400
    df=pd.DataFrame([data]) # converting json data to pandas dataframe 
    #but why [data]? because model.predict() method accepts 2D array or dataframe
    probability=model.predict_proba(df)[0][1].item() #first 0 for first row, second 1 for probability of class 1(habitable
    prediction=int(probability>=0.3) #doing predction based on probability obtained
    return jsonify({
        "habitable":prediction,
        "probability":round(probability,3)
    })
    
@app.route('/rank',methods=["POST"])
def rank():
    data=request.get_json()
    df=pd.DataFrame(data)
    df['habitability_probability']=model.predict_proba(df)[:,1] .tolist() #[:,1] to get probability of class 1 for all rows
    #creating probability column in dataframe for each exoplanet
    df["rank"]=df['habitability_probability'].rank(ascending=False)#creating rank column based on probability and assigning rank in descending order
    df=df.sort_values(by='rank')# sorting dataframe based on rank
    return jsonify({
        "ranked_exoplanets":df.to_dict(orient="records") #df.to.dict means converting dataframe to dictionary
        #why dictionary because it has many rows and each record is a planet's data
        #so we send it as list of dictionaries(records) in json format. Example:[{"planet1_data"},{...}],say planet1_data has keys like mass,radius,etc
    })
    
@app.route('/predict_input',methods=["POST"])
def predict_input():
    data=request.get_json()  #getting data from form submitted by user
    required_fields=[
        'radius',
        'mass',
        'temp',
        'orbital_period',
        'distance_star',
        'star_temp',
        'eccentricity',
        'semi_major_axis',
        'star_type'
    ]
    
    missing=[field for field in required_fields if field not in data]
    if missing:
        return jsonify({
            "error":"Missing fields",
            "missing_fields":missing
        }),400
    return jsonify({
        "message":"Input submitted successfully",
        "input_data":data
    }),200
    
@app.route("/planets", methods=["GET"])
def get_planets():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM exoplanets LIMIT 10;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return jsonify({
        "planets": rows
    })

if __name__=="__main__":
    app.run(debug=True)