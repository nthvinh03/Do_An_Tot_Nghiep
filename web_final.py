import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

# Thiết lập tiêu đề trang
st.set_page_config(layout="centered", page_title="Hệ thống dự đoán bệnh tiểu đường")
st.title('Hệ thống dự đoán bệnh tiểu đường')

st.write("""
**Ứng dụng này dự đoán nguy cơ mắc bệnh tiểu đường dựa trên các thông số sức khỏe của bệnh nhân.
Vui lòng nhập các thông tin sau**
""")

# Xác định thứ tự các đặc trưng để khớp với dữ liệu huấn luyện
# Removed 'Education' and 'Income' from feature_order
feature_order = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                 'HvyAlcoholConsump', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
                 'Age']

# Hàm tạo giao diện nhập liệu
def user_input_features():
    st.subheader("Thông tin sức khỏe")

    # Group binary radio buttons into pairs using columns for better alignment
    col_radio1, col_radio2 = st.columns(2)
    with col_radio1:
        HighBP = st.radio(
            'Bệnh nhân từng được thông báo là bị **cao huyết áp**?',
            ('Không', 'Có'),
            help='Bệnh nhân từng được bác sĩ, y tá hoặc chuyên gia y tế thông báo là bị cao huyết áp',
            key='HighBP_radio'
        )
    with col_radio2:
        HighChol = st.radio(
            'Bệnh nhân từng được thông báo là bị **cholesterol cao**?',
            ('Không', 'Có'),
            help='Bệnh nhân từng được bác sĩ, y tá hoặc chuyên gia y tế thông báo là bị cholesterol cao',
            key='HighChol_radio'
        )

    col_radio3, col_radio4 = st.columns(2)
    with col_radio3:
        CholCheck = st.radio(
            'Đã kiểm tra **cholesterol** trong vòng 5 năm qua?',
            ('Không', 'Có'),
            key='CholCheck_radio'
        )
    with col_radio4:
        BMI = st.number_input(
            'Chỉ số khối cơ thể (BMI)',
            min_value=10.0, max_value=60.0, value=22.0,
            help='Chỉ số khối cơ thể (Body Mass Index)',
            key='BMI_input'
        )

    col_radio5, col_radio6 = st.columns(2)
    with col_radio5:
        Smoker = st.radio(
            'Bệnh nhân đã từng **hút ít nhất 100 điếu thuốc** trong đời?',
            ('Không', 'Có'),
            help='Lưu ý: 5 gói = 100 điếu thuốc',
            key='Smoker_radio'
        )
    with col_radio6:
        Stroke = st.radio(
            'Bệnh nhân đã từng được thông báo là bị **đột quỵ**?',
            ('Không', 'Có'),
            key='Stroke_radio'
        )

    col_radio7, col_radio8 = st.columns(2)
    with col_radio7:
        HeartDiseaseorAttack = st.radio(
            'Từng được thông báo mắc **bệnh tim mạch vành (CHD) hoặc nhồi máu cơ tim (MI)**?',
            ('Không', 'Có'),
            key='HeartDiseaseorAttack_radio'
        )
    with col_radio8:
        PhysActivity = st.radio(
            'Trong 30 ngày qua, Bệnh nhân có **hoạt động thể chất** ngoài công việc hàng ngày?',
            ('Không', 'Có'),
            help='Hoạt động thể chất hoặc tập thể dục ngoài công việc hàng ngày',
            key='PhysActivity_radio'
        )

    col_radio9, col_radio10 = st.columns(2)
    with col_radio9:
        Fruits = st.radio(
            'Tiêu thụ **trái cây** 1 lần trở lên mỗi ngày?',
            ('Không', 'Có'),
            key='Fruits_radio'
        )
    with col_radio10:
        Veggies = st.radio(
            'Tiêu thụ **rau củ** 1 lần trở lên mỗi ngày?',
            ('Không', 'Có'),
            key='Veggies_radio'
        )

    col_radio11, col_radio12 = st.columns(2)
    with col_radio11:
        HvyAlcoholConsump = st.radio(
            'Tiêu thụ **rượu ở mức nặng**?',
            ('Không', 'Có'),
            help='Nam uống >14 ly/tuần, nữ >7 ly/tuần',
            key='HvyAlcoholConsump_radio'
        )
    with col_radio12:
        DiffWalk = st.radio(
            'Bệnh nhân có gặp **khó khăn nghiêm trọng khi đi bộ hoặc leo cầu thang**?',
            ('Không', 'Có'),
            key='DiffWalk_radio'
        )

    st.markdown("---") # Separator
    st.subheader("Đánh giá sức khỏe tổng thể")

    # Define the mapping for GenHlth labels (moved outside the column block)
    gen_hlth_labels = {1: 'Rất tốt', 2: 'Tốt', 3: 'Khá tốt', 4: 'Kém', 5: 'Rất kém'}

    # Use columns for these sliders
    col_health_slider1, col_health_slider2, col_health_slider3 = st.columns(3)

    with col_health_slider1:
        # Use st.slider and dynamically update its label
        # Get the current selected value (or default)
        GenHlth_val = st.session_state.get('GenHlth_slider_key', 3) 
        current_label = gen_hlth_labels[GenHlth_val]

        GenHlth = st.slider(
            f'Đánh giá chung về tình trạng sức khỏe: **{current_label}**',
            min_value=1, max_value=5, value=GenHlth_val,
            step=1, # Ensure it selects integers
            key='GenHlth_slider_key' # Important: use a unique key for the slider
        )

    with col_health_slider2:
        MentHlth = st.slider(
            'Số ngày tâm lý không ổn định trong 30 ngày qua',
            0, 30, 0,
            help='Số ngày cảm thấy căng thẳng, trầm cảm, v.v.',
            key='MentHlth_slider'
        )

    with col_health_slider3:
        PhysHlth = st.slider(
            'Số ngày sức khỏe thể chất không tốt trong 30 ngày qua',
            0, 30, 0,
            help='Số ngày cảm thấy ốm hoặc bị chấn thương',
            key='PhysHlth_slider'
        )

    st.markdown("---") # Separator
    st.subheader("Thông tin cá nhân")

    # Using columns for demographic info for better alignment
    col_demog1 = st.columns(1)[0] 

    with col_demog1:
        # Nhóm tuổi
        age_options = {
            '1': '18-24 tuổi',
            '2': '25-29 tuổi',
            '3': '30-34 tuổi',
            '4': '35-39 tuổi',
            '5': '40-44 tuổi',
            '6': '45-49 tuổi',
            '7': '50-54 tuổi',
            '8': '55-59 tuổi',
            '9': '60-64 tuổi',
            '10': '65-69 tuổi',
            '11': '70-74 tuổi',
            '12': '75-79 tuổi',
            '13': '80+ tuổi'
        }
        age_choice = st.selectbox('Nhóm tuổi', list(age_options.values()), key='Age_selectbox')
        Age = list(age_options.keys())[list(age_options.values()).index(age_choice)]

    # Removed Education and Income input fields

    # Chuyển đổi câu trả lời Có/Không thành 1/0
    binary_features = {
        'HighBP': 1 if HighBP == 'Có' else 0,
        'HighChol': 1 if HighChol == 'Có' else 0,
        'CholCheck': 1 if CholCheck == 'Có' else 0,
        'Smoker': 1 if Smoker == 'Có' else 0,
        'Stroke': 1 if Stroke == 'Có' else 0,
        'HeartDiseaseorAttack': 1 if HeartDiseaseorAttack == 'Có' else 0,
        'PhysActivity': 1 if PhysActivity == 'Có' else 0,
        'Fruits': 1 if Fruits == 'Có' else 0,
        'Veggies': 1 if Veggies == 'Có' else 0,
        'HvyAlcoholConsump': 1 if HvyAlcoholConsump == 'Có' else 0,
        'DiffWalk': 1 if DiffWalk == 'Có' else 0
    }

    # Tạo từ điển với tất cả các đặc trưng
    features = {
        **binary_features,
        'BMI': BMI,
        'GenHlth': int(GenHlth), # Use the GenHlth value from the slider
        'MentHlth': int(MentHlth),
        'PhysHlth': int(PhysHlth),
        'Age': int(Age),
        # Removed 'Education' and 'Income' from features dictionary
    }

    # Trả về các đặc trưng theo đúng thứ tự
    ordered_features = {feature: features[feature] for feature in feature_order}
    return ordered_features

# Thu thập đầu vào từ người dùng
user_inputs = user_input_features()

# ---
st.markdown("---") # Separator

# Hiển thị thông số đã chọn
st.subheader('Thông số bệnh nhân đã nhập:')

# Tạo DataFrame để hiển thị với các nhãn dễ đọc
display_df = pd.DataFrame(user_inputs, index=['Giá trị'])


# Tạo từ điển mapping để hiển thị giá trị dễ đọc
mapping_dict = {
    'HighBP': {0: 'Không', 1: 'Có'},
    'HighChol': {0: 'Không', 1: 'Có'},
    'CholCheck': {0: 'Không', 1: 'Có'},
    'Smoker': {0: 'Không', 1: 'Có'},
    'Stroke': {0: 'Không', 1: 'Có'},
    'HeartDiseaseorAttack': {0: 'Không', 1: 'Có'},
    'PhysActivity': {0: 'Không', 1: 'Có'},
    'Fruits': {0: 'Không', 1: 'Có'},
    'Veggies': {0: 'Không', 1: 'Có'},
    'HvyAlcoholConsump': {0: 'Không', 1: 'Có'},
    'DiffWalk': {0: 'Không', 1: 'Có'},
    'GenHlth': {1: 'Rất tốt', 2: 'Tốt', 3: 'Khá tốt', 4: 'Kém', 5: 'Rất kém'},
}

# Áp dụng mapping
for col in display_df.columns:
    if col in mapping_dict:
        display_df[col] = display_df[col].map(mapping_dict[col])



# Đổi tên các cột để dễ đọc hơn
column_names = {
    'HighBP': 'Cao huyết áp',
    'HighChol': 'Cholesterol cao',
    'CholCheck': 'Kiểm tra cholesterol (5 năm)',
    'BMI': 'Chỉ số BMI',
    'Smoker': 'Hút thuốc',
    'Stroke': 'Đột quỵ',
    'HeartDiseaseorAttack': 'Bệnh tim/Đau tim',
    'PhysActivity': 'Hoạt động thể chất',
    'Fruits': 'Ăn trái cây hàng ngày',
    'Veggies': 'Ăn rau củ hàng ngày',
    'HvyAlcoholConsump': 'Uống nhiều rượu',
    'GenHlth': 'Sức khỏe tổng quát',
    'MentHlth': 'Ngày sức khỏe tâm thần kém',
    'PhysHlth': 'Ngày sức khỏe thể chất kém',
    'DiffWalk': 'Khó khăn khi đi bộ',
    'Age': 'Nhóm tuổi',
    # Removed 'Education' and 'Income' from column_names
}
display_df = display_df.rename(columns=column_names)

# Hiển thị DataFrame dưới dạng bảng dọc (transpose)
final_display_df = pd.DataFrame({
    'Thông số': display_df.T.index,
    'Trạng thái': display_df.T.iloc[:, 0].values
})
st.dataframe(final_display_df, use_container_width=True,hide_index=True) # use_container_width makes it span the full width

# ---
st.markdown("---") # Separator

# Hàm để tải mô hình và scaler
@st.cache_resource
def load_model_components():
    try:
        model = load_model('models/best_model.h5')
        with open('models/diabetes_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình hoặc scaler. Đảm bảo các tệp có sẵn trong thư mục 'models'. Lỗi: {str(e)}")
        return None, None

# Nút để thực hiện dự đoán
if st.button('Dự đoán nguy cơ tiểu đường', type='primary'):
    # Tải mô hình và scaler
    model, scaler = load_model_components()

    if model is not None and scaler is not None:
        try:
            # Chuyển đổi đầu vào thành mảng numpy theo đúng thứ tự
            input_array = np.array([list(user_inputs.values())])

            # Chuẩn hóa dữ liệu đầu vào
            input_scaled = scaler.transform(input_array)

            input_for_MLP=input_scaled

            # Thực hiện dự đoán
            prediction = model.predict(input_for_MLP)
            prediction_probability = float(prediction[0][0])

            # Hiển thị kết quả
            st.subheader('Kết quả dự đoán:')

            # Tạo thanh progress
            progress_color = 'green' if prediction_probability < 0.5 else 'red'
            st.markdown(
                f"""
                <style>
                    .stProgress > div > div > div > div {{
                        background-color: {progress_color};
                    }}
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.progress(prediction_probability)

            if prediction_probability >= 0.5:
                st.error(f"⚠️ **Nguy cơ cao mắc bệnh tiểu đường** với xác suất **{prediction_probability:.2%}**")
                st.markdown("""
                **Khuyến nghị:**
                - Nguy cơ mắc bệnh cao.
                - Nên tới Trung tâm y tế, Bệnh viện để kiểm tra chuyên sâu và được bác sĩ, chuyên gia chuẩn đoán kĩ.
                - Cân nhắc thay đổi lối sống và chế độ ăn uống.
                - Kiểm tra đường huyết định kỳ.
                """)
            else:
                st.success(f"✅ **Nguy cơ thấp mắc bệnh tiểu đường** với xác suất **{prediction_probability:.2%}**")
                st.markdown("""
                **Khuyến nghị:**
                - Nguy cơ mắc bệnh thấp.
                - Nên duy trì lối sống lành mạnh.
                - Kiểm tra sức khỏe định kỳ.
                - Giữ cân nặng hợp lý. 
                - Nên duy trì các hoạt động thể chất, tập thể dục, thể thao thường xuyên.
                """)

            # Hiển thị trực quan kết quả
            st.subheader("Biểu đồ xác suất")
            fig, ax = plt.subplots(figsize=(8, 2))
            bars = ax.barh(['Không mắc tiểu đường', 'Mắc tiểu đường'],
                            [1-prediction_probability, prediction_probability])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Xác suất')
            ax.set_title('Dự đoán nguy cơ tiểu đường')

            # Thêm màu sắc cho các cột
            bars[0].set_color('green')
            bars[1].set_color('red')

            # Thêm nhãn văn bản cho các cột
            ax.text(max(0.05, (1-prediction_probability)/2), 0, f"{(1-prediction_probability):.2%}",
                                va='center', ha='center', color='black')
            ax.text(max(0.05, prediction_probability/2), 1, f"{prediction_probability:.2%}",
                                va='center', ha='center', color='white' if prediction_probability > 0.3 else 'black')

            st.pyplot(fig)
            plt.close(fig) # Close the figure to free memory

            # Hiển thị các yếu tố nguy cơ
            st.subheader("Các yếu tố nguy cơ chính được phát hiện:")
            risk_factors = []

            if user_inputs['HighBP'] == 1:
                risk_factors.append("- **Cao huyết áp**")
            if user_inputs['HighChol'] == 1:
                risk_factors.append("- **Cholesterol cao**")
            if user_inputs['BMI'] >= 25 and user_inputs['BMI'] < 30:
                risk_factors.append("- **BMI cao** (Béo phì độ I)")
            if user_inputs['BMI'] >=30 :
                risk_factors.append("- **BMI cao** (Béo phì độ II)")
            if user_inputs['HeartDiseaseorAttack'] == 1:
                risk_factors.append("- **Tiền sử bệnh tim**")
            if user_inputs['Smoker'] == 1:
                risk_factors.append("- **Hút thuốc**")
            if user_inputs['GenHlth'] >= 4:
                risk_factors.append("- **Sức khỏe tổng thể kém**")
            if user_inputs['Age'] >= 8: # Older age group (55+)
                risk_factors.append("- **Tuổi cao**")
            if user_inputs['DiffWalk'] == 1:
                risk_factors.append("- **Khó khăn khi đi bộ**")

            if risk_factors:
                for factor in risk_factors:
                    st.markdown(factor)
            else:
                st.write("Không phát hiện yếu tố nguy cơ đáng kể dựa trên thông tin đã nhập.")

        except Exception as e:
            st.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
    else:
        st.info("""
        Không tìm thấy tệp mô hình và/hoặc scaler. Đảm bảo bạn đã huấn luyện và lưu chúng vào thư mục `models`.

        Để lưu mô hình và scaler sau khi huấn luyện, hãy sử dụng:
        ```python
        # Lưu scaler
        with open('models/diabetes_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # Lưu mô hình
        model.save('models/best_model.h5')
        ```
        """)

# ---
st.markdown("---") # Separator

# Hiển thị giải thích về các trường dữ liệu
with st.expander("Giải thích về các trường dữ liệu"):
    st.markdown("""
    ### **Giải nghĩa các trường dữ liệu:**

    - **Cao huyết áp (HighBP)**: Bệnh nhân từng được bác sĩ, y tá hoặc chuyên gia y tế thông báo là bị cao huyết áp (0: Không, 1: Có).

    - **Cholesterol cao (HighChol)**: Bệnh nhân từng được bác sĩ, y tá hoặc chuyên gia y tế thông báo là bị cholesterol cao (0: Không, 1: Có).

    - **Kiểm tra cholesterol (CholCheck)**: Đã kiểm tra cholesterol trong vòng 5 năm qua (0: Không, 1: Có).

    - **BMI**: Chỉ số khối cơ thể (Body Mass Index)(Tính bằng công thức : BMI = (Cân nặng/ Chiều cao^2)).

    - **Hút thuốc (Smoker)**: Bệnh nhân đã từng hút ít nhất 100 điếu thuốc trong đời (Lưu ý: 5 gói = 100 điếu thuốc) (0: Không, 1: Có).

    - **Đột quỵ (Stroke)**: Bệnh nhân đã từng được thông báo là bị đột quỵ (0: Không, 1: Có).

    - **Bệnh tim/Đau tim (HeartDiseaseorAttack)**: Từng được thông báo mắc bệnh tim mạch vành (CHD) hoặc nhồi máu cơ tim (MI) (0: Không, 1: Có).

    - **Hoạt động thể chất (PhysActivity)**: Trong 30 ngày qua, Bệnh nhân có hoạt động thể chất hoặc tập thể dục nào ngoài công việc hàng ngày không? (0: Không, 1: Có).

    - **Ăn trái cây (Fruits)**: Tiêu thụ trái cây 1 lần trở lên mỗi ngày (0: Không, 1: Có).

    - **Ăn rau củ (Veggies)**: Tiêu thụ rau củ 1 lần trở lên mỗi ngày (0: Không, 1: Có).

    - **Uống nhiều rượu (HvyAlcoholConsump)**: Người uống rượu nặng (nam uống >14 ly/tuần, nữ >7 ly/tuần) (0: Không, 1: Có).

    - **Sức khỏe tổng quát (GenHlth)**: Đánh giá chung về tình trạng sức khỏe của Bệnh nhân (1: Rất tốt ~ 5: Rất kém).

    - **Sức khỏe tâm thần (MentHlth)**: Trong 30 ngày qua, Bệnh nhân cảm thấy tâm lý không ổn định (căng thẳng, trầm cảm, v.v.) trong bao nhiêu ngày? (0 ~ 30 ngày).

    - **Sức khỏe thể chất (PhysHlth)**: Trong 30 ngày qua, sức khỏe thể chất không tốt (ốm, chấn thương) trong bao nhiêu ngày? (0 ~ 30 ngày).

    - **Khó khăn khi đi bộ (DiffWalk)**: Bệnh nhân có gặp khó khăn nghiêm trọng khi đi bộ hoặc leo cầu thang không? (0: Không, 1: Có).

    - **Nhóm tuổi (Age)**: Nhóm tuổi chia thành 13 mức (1 ~ 13). 1 tương ứng 18-24 tuổi, 13 tương ứng 80+ tuổi (mỗi 5 tuổi là 1 mức).
    """)

st.markdown("""
<div style="text-align: center;">

**Lưu ý:** Công cụ này chỉ dành cho mục đích giáo dục và nghiên cứu, không thay thế cho tư vấn y tế chuyên nghiệp.
Vui lòng tham khảo ý kiến bác sĩ để được chẩn đoán và điều trị chính xác.
</div>
""", unsafe_allow_html=True)