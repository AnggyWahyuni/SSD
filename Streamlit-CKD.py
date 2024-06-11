import pickle
import streamlit as st
import numpy as np

# Membaca model
try:
    with open('CKD_RFClassifier.sav', 'rb') as model_file:
        CKD_model = pickle.load(model_file)
    with open('standardscaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    st.error(f'Error dalam memuat model atau scaler: {e}')
    st.stop()

# Mapping kategori ke keterangan
PusCellClumps_mapping = {'notpresent': 'Not Present', 'present': 'Present'}
Bacteria_mapping = {'notpresent': 'Not Present', 'present': 'Present'}
Hypertension_mapping = {'no': 'No', 'yes': 'Yes'}
DiabetesMellitus_mapping = {'no': 'No', 'yes': 'Yes'}
CoronaryArteryDisease_mapping = {'no': 'No', 'yes': 'Yes'}
Appetite_mapping = {'good': 'Good', 'poor': 'Poor'}
PedalEdema_mapping = {'no': 'No', 'yes': 'Yes'}
Anemia_mapping = {'no': 'No', 'yes': 'Yes'}

# Judul web
st.title('CKD Prediction')

# Mengambil input dari user
col1, col2 = st.columns(2)
with col1:
    Age = st.text_input('Age (years)')
    BloodPressure = st.text_input('Blood Pressure (Diastolic)')
    BloodGlucoseRandom = st.text_input('Blood Glucose Random (mgs/dL)')
    BloodUrea = st.text_input('Blood Urea (mgs/dL)')
    SerumCreatinine = st.text_input('Serum Creatinine (mgs/dL)')
    Sodium = st.text_input('Sodium (mEq/L)')
    Potassium = st.text_input('Potassium (mEq/L)')
    Hemoglobin = st.text_input('Hemoglobin (g/L)')
    PackedCellVolume = st.text_input('PCV (%)')
    RedBloodCount = st.text_input('RBCC (million/cumm)')
with col2:
    WhiteBloodCount = st.text_input('WBCC (cells/cumm)')
    PusCellClumps = st.selectbox('Pus Cell Clumps', list(PusCellClumps_mapping.values()))
    Bacteria = st.selectbox('Bacteria', list(Bacteria_mapping.values()))
    Hypertension = st.selectbox('Hypertension', list(Hypertension_mapping.values()))
    DiabetesMellitus = st.selectbox('Diabetes Mellitus', list(DiabetesMellitus_mapping.values()))
    CoronaryArteryDisease = st.selectbox('Coronary Artery Disease', list(CoronaryArteryDisease_mapping.values()))
    Appetite = st.selectbox('Appetite', list(Appetite_mapping.values()))
    PedalEdema = st.selectbox('Pedal Edema', list(PedalEdema_mapping.values()))
    Anemia = st.selectbox('Anemia', list(Anemia_mapping.values()))

# Membuat tombol prediksi
if st.button('Prediction Result'):
    try:
        # Konversi input yang relevan menjadi float
        Age = int(Age)
        BloodPressure = int(BloodPressure)
        BloodGlucoseRandom = int(BloodGlucoseRandom)
        BloodUrea = int(BloodUrea)
        SerumCreatinine = float(SerumCreatinine)
        Sodium = int(Sodium)
        Potassium = float(Potassium)
        Hemoglobin = float(Hemoglobin)
        PackedCellVolume = int(PackedCellVolume)
        RedBloodCount = float(RedBloodCount)
        WhiteBloodCount = int(WhiteBloodCount)

        # One-hot encoding untuk input kategori
        PusCellClumps_encoded = [0] * 2
        Bacteria_encoded = [0] * 2
        Hypertension_encoded = [0] * 2
        DiabetesMellitus_encoded = [0] * 2
        CoronaryArteryDisease_encoded = [0] * 2
        Appetite_encoded = [0] * 2
        PedalEdema_encoded = [0] * 2
        Anemia_encoded = [0] * 2

        # Mapping input kategori ke index yang sesuai
        PusCellClumps_encoded[list(PusCellClumps_mapping.values()).index(PusCellClumps)] = 1 
        Bacteria_encoded[list(Bacteria_mapping.values()).index(Bacteria)] = 1
        Hypertension_encoded[list(Hypertension_mapping.values()).index(Hypertension)] = 1
        DiabetesMellitus_encoded[list(DiabetesMellitus_mapping.values()).index(DiabetesMellitus)] = 1
        CoronaryArteryDisease_encoded[list(CoronaryArteryDisease_mapping.values()).index(CoronaryArteryDisease)] = 1
        Appetite_encoded[list(Appetite_mapping.values()).index(Appetite)] = 1
        PedalEdema_encoded[list(PedalEdema_mapping.values()).index(PedalEdema)] = 1
        Anemia_encoded[list(Anemia_mapping.values()).index(Anemia)] = 1
        
        # Hanya skalakan input numerik
        numeric_features = scaler.transform([[Age, BloodPressure, BloodGlucoseRandom, BloodUrea, SerumCreatinine, Sodium, Potassium, Hemoglobin, PackedCellVolume, RedBloodCount, WhiteBloodCount]])
        
        # Gabungkan fitur numerik yang telah diskalakan dengan fitur kategorikal yang telah di-one-hot encoding
        input_data_scaled = list(numeric_features[0]) + PusCellClumps_encoded + Bacteria_encoded + Hypertension_encoded + DiabetesMellitus_encoded + CoronaryArteryDisease_encoded + Appetite_encoded + PedalEdema_encoded + Anemia_encoded
        
        # Lakukan prediksi
        CKD_prediction = CKD_model.predict([input_data_scaled])

        # Tentukan hasil prediksi
        if np.array_equal(CKD_prediction[0], [1, 0]):
            CKD_class = 'NOT CKD'
        else:
            CKD_class = 'CKD'

        st.success(CKD_class)
    except ValueError as e:
        st.error(f'Error dalam konversi input: {e}')
    except Exception as e:
        st.error(f'Error dalam prediksi: {e}')
