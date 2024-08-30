import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.preprocessing import LabelEncoder

# モデルの読み込み
with open('lgbm_model.pkl', 'rb') as model_file:
    lgbm_model = pickle.load(model_file)

# Streamlitアプリの構築
st.title('機械学習によるOFC結果予測アプリ')

# ユーザー入力の取得
def user_input_features():
    # 基本情報
    st.header('基本情報')
    gender = st.selectbox('性別', ['男', '女'])
    age_at_test = st.number_input('検査時の年齢', 0, 18, 0)  # 年齢をnumber_inputに変更
    
    # アレルギー関連情報
    st.header('アレルギー関連情報')
    atopic_dermatitis = st.selectbox('アトピー性皮膚炎の既往', ['はい', 'いいえ'])
    bronchial_asthma = st.selectbox('気管支喘息の既往', ['はい', 'いいえ'])
    allergic_rhinitis = st.selectbox('アレルギー性鼻炎の既往', ['はい', 'いいえ'])
    past_allergic_symptoms = st.selectbox('負荷食品による過去の明らかなアレルギー症状誘発歴', ['はい', 'いいえ'])
    other_food_allergy = st.selectbox('負荷食品以外の食品による食物アレルギー', ['はい', 'いいえ'])

    # 血液検査データ
    st.header('血液検査データ')
    total_IgE = st.number_input('総IgE（IU/ml）', 0.0, 1000.0, 100.0)
    WBC = st.number_input('白血球数（WBC）（/μL）', 0, 10000, 5000)
    eosinophils = st.number_input('好酸球(%)', 0.0, 100.0, 5.0)  # 好酸球をnumber_inputに変更
    coarse_antigen_sIgE = st.number_input('粗抗原sIgE（UA/ml）', 0.0, 100.0, 1.0)

    # 負荷試験関連情報
    st.header('負荷試験関連情報')
    final_total_intake = st.number_input('最終総摂取量', 0.0, 100.0, 50.0)
    planned_total_load = st.number_input('予定総負荷量', 0.0, 100.0, 50.0)

    # 入力をDataFrameにまとめる
    data = {
        'gender': gender,
        'atopic_dermatitis': atopic_dermatitis,
        'bronchial_asthma': bronchial_asthma,
        'allergic_rhinitis': allergic_rhinitis,
        'age_at_test': age_at_test,
        'total_IgE': total_IgE,
        'WBC': WBC,
        'eosinophils': eosinophils,
        'coarse_antigen_sIgE': coarse_antigen_sIgE,
        'past_allergic_symptoms': past_allergic_symptoms,
        'other_food_allergy': other_food_allergy,
        'final_total_intake': final_total_intake,
        'planned_total_load': planned_total_load
    }
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 入力されたデータのラベルエンコーディング
for col in input_df.select_dtypes(include='object').columns:
    input_df[col] = LabelEncoder().fit_transform(input_df[col].astype(str))

st.subheader('入力されたデータ')
st.write(input_df)

# ボタンを追加して予測を実行
if st.button('予測する'):
    # 入力データのラベルエンコーディング（必要な場合）
    input_df_encoded = input_df.copy()
    label_encoders = {}
    for column in input_df_encoded.select_dtypes(include='object').columns:
        label_encoders[column] = pd.factorize(input_df_encoded[column])[0]
        input_df_encoded[column] = label_encoders[column]

    # モデルを使用して予測
    prediction = (lgbm_model.predict(input_df_encoded) > 0.5).astype(int)

    # 予測結果の表示
    result = '陽性' if prediction[0] == 1 else '陰性'
    st.subheader('予測結果')
    st.write(result)