import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('Best_XGBoost_Model.pkl')

# Streamlit app layout
st.set_page_config(page_title="Rice Yield Predictor 🌾", page_icon="🌾")

st.title('🌾 Rice Yield Prediction Under Hybrid CO₂ Conditions')
st.subheader('🚀 Powered by Machine Learning (XGBoost)')

st.markdown("""
This application predicts rice grain yield based on phenological parameters and CO₂ experimental conditions.

**Upload CSV file with these columns:**
- Physiological_Maturity
- Panicle_Initiation
- Runs
- Anthesis
- Condition_Encoded (0 = Current CO₂, 1 = Projected CO₂)
""")

uploaded_file = st.file_uploader("📄 Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.success('✅ File uploaded successfully!')

    st.write('📈 Preview of Uploaded Data:')
    st.dataframe(input_data)

    # 🛡️ Fix columns and enforce numeric types
    input_data = input_data[['Physiological_Maturity', 'Panicle_Initiation', 'Runs', 'Anthesis', 'Condition_Encoded']]
    input_data = input_data.apply(pd.to_numeric)

    # 🔥 Predict
    st.subheader('🔮 Predicted Rice Grain Yield (kg/ha)')

    if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.success('✅ File uploaded successfully!')

    st.write('📈 Preview of Uploaded Data:')
    st.dataframe(input_data)

    # 🛡️ STEP 1: Select and fix the order of columns
    input_data_features = input_data[['Physiological_Maturity', 'Panicle_Initiation', 'Runs', 'Anthesis', 'Condition_Encoded']]

    # 🛡️ STEP 2: Force columns to numeric
    input_data_features = input_data_features.apply(pd.to_numeric)

    # 🔮 STEP 3: Predict using model
    predictions = model.predict(input_data_features)

    # 🛡️ STEP 4: Attach predictions to original input_data
    input_data['Predicted_Grain_Yield_kg/ha'] = predictions

    # 📈 STEP 5: Display updated input_data
    st.subheader('🔮 Predicted Rice Grain Yield (kg/ha)')
    st.dataframe(input_data)

    # 📥 STEP 6: Download predictions
    csv = input_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prediction Results",
        data=csv,
        file_name='Predicted_Rice_Yield.csv',
        mime='text/csv'
    )

