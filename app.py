import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import Row

# --------------------------------------------------
# 1. START SPARK (LOCAL MODE)
# --------------------------------------------------
@st.cache_resource
def start_spark():
    return SparkSession.builder \
        .appName("Diabetes Prediction App") \
        .master("local[*]") \
        .getOrCreate()

spark = start_spark()

# --------------------------------------------------
# 2. LOAD SAVED PIPELINE MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return PipelineModel.load("diabetes_pipeline_model")

model = load_model()

# --------------------------------------------------
# 3. STREAMLIT UI
# --------------------------------------------------
st.title("🩺 Diabetes Prediction App (Spark ML)")

st.write("Enter patient details:")

Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
Glucose = st.number_input("Glucose", min_value=0.0)
BloodPressure = st.number_input("Blood Pressure", min_value=0.0)
SkinThickness = st.number_input("Skin Thickness", min_value=0.0)
Insulin = st.number_input("Insulin", min_value=0.0)
BMI = st.number_input("BMI", min_value=0.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0)
Age = st.number_input("Age", min_value=1, step=1)

# --------------------------------------------------
# 4. PREDICTION
# --------------------------------------------------
if st.button("Predict"):
    input_row = Row(
        Pregnancies=Pregnancies,
        Glucose=Glucose,
        BloodPressure=BloodPressure,
        SkinThickness=SkinThickness,
        Insulin=Insulin,
        BMI=BMI,
        DiabetesPedigreeFunction=DiabetesPedigreeFunction,
        Age=Age,
        Outcome=0  # dummy label (required by pipeline)
    )

    input_df = spark.createDataFrame([input_row])

    prediction = model.transform(input_df)

    result = prediction.select("prediction", "probability").collect()[0]

    st.subheader("Result")
    if result["prediction"] == 1.0:
        st.error(f"Diabetes Detected (Probability: {result['probability'][1]:.2f})")
    else:
        st.success(f"No Diabetes (Probability: {result['probability'][0]:.2f})")
