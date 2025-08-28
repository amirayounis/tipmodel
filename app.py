import streamlit as st
import pickle as pkl
import numpy as np

lr = pkl.load(open("linear_model.pkl", "rb"))
ohe = pkl.load(open("ohe.pkl", "rb"))
st.header("My First Streamlit App")
st.subheader("Hello, world!")
results = {}
with st.form("my_form"):
    st.write("Inside the form")
    day_val = st.selectbox("Day", ['Sat', 'Sun', 'Thur', 'Fri'])
    time_val= st.selectbox("Time", ['Dinner', 'Lunch'])
    gender_val=st.radio("Gender", ['Male', 'Female'])
    is_smoker = st.checkbox("smoker?")
    total_bill_val = st.text_input("Total bill")
    size = st.number_input("Size of the group", min_value=1, max_value=6, step=1)
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        results = {
            "Day": day_val,
            "Time": time_val,
            "Gender": gender_val,
            "Smoker": 'Yes' if is_smoker else 'No',
            "Total bill": total_bill_val,
            "Size": size
        }
if results:
    st.write(results)
    st.write("your approximate tip is")
    category = ohe.transform(np.array([[gender_val,results["Smoker"],day_val, time_val]]))
    x=np.concatenate(( np.array([[float(total_bill_val), size]]),category), axis=1)
    y=lr.predict(x)
    st.write(y)
