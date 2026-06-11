# THÀNH PHẦN KHỞI TẠO ĐẦU TIÊN
import streamlit as st
st.set_page_config(layout="wide", page_title="Fraud Detection App", page_icon="🕵️‍♂️")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ==========================================
# CÁC HÀM DÙNG CHUNG (CACHE)
# ==========================================
@st.cache_data
def load_data(file_bytes, file_name):
    """Đọc dữ liệu từ file upload (bytes) để cache hợp lệ."""
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_bytes)
    elif file_name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_bytes)
    else:
        return None
    return df

# Cấu hình các biến đầu vào dựa trên tập dataset1.csv
FEATURES = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
TARGET = 'isFraud'

# ==========================================
# THÀNH PHẦN 1: SIDEBAR - CẤU HÌNH BỀN VỮNG
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    uploaded_file = st.file_uploader("Tải lên file dữ liệu (.csv, .xlsx)", type=['csv', 'xlsx'])
    
    st.divider()
    
    model_choice = st.selectbox(
        "Lựa chọn Mô hình",
        options=["Random Forest", "Logistic Regression"],
        help="Thuật toán học máy dùng để phân loại giao dịch."
    )
    
    st.subheader("Tham số mô hình AI")
    if model_choice == "Random Forest":
        n_estimators = st.slider("Số lượng cây (n_estimators)", min_value=10, max_value=200, value=50, step=10, help="Số lượng cây quyết định trong rừng.")
        max_depth = st.slider("Độ sâu tối đa (max_depth)", min_value=2, max_value=20, value=10, step=1, help="Độ sâu của mỗi cây để tránh quá khớp (overfitting).")
        random_state = st.number_input("Random State", value=42, step=1)
    else:
        max_iter = st.number_input("Số vòng lặp tối đa (max_iter)", min_value=100, max_value=1000, value=200, step=50)
        C_param = st.slider("Tham số điều chuẩn (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
        random_state = st.number_input("Random State", value=42, step=1)
    
    test_size = st.slider("Tỷ lệ tập kiểm định (test_size)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    st.divider()
    train_button = st.button("🚀 Huấn luyện Mô hình", type="primary", use_container_width=True)

# ==========================================
# THÀNH PHẦN 2: HEADER - VÙNG ĐỊNH HƯỚNG
# ==========================================
st.title("🕵️‍♂️ Ứng dụng Phát hiện Giao dịch Gian lận")
st.caption("Ứng dụng AI phân tích và phát hiện các giao dịch tài chính bất thường dựa trên lịch sử giao dịch và biến động số dư tài khoản.")

if uploaded_file is None:
    st.info("👈 Vui lòng tải lên file dữ liệu (ví dụ: dataset1.csv) ở thanh bên trái để bắt đầu.")
    st.stop()

# Đọc dữ liệu
try:
    df_raw = load_data(uploaded_file, uploaded_file.name)
    if df_raw is None:
        st.error("Định dạng file không được hỗ trợ!")
        st.stop()
except Exception as e:
    st.error(f"Lỗi đọc file: {e}")
    st.stop()

# Kiểm tra schema dữ liệu
missing_cols = [col for col in FEATURES + [TARGET] if col not in df_raw.columns]
if missing_cols:
    st.error(f"File tải lên thiếu các cột bắt buộc: {', '.join(missing_cols)}")
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}** | 📊 Kích thước: {df_raw.shape[0]:,} dòng, {df_raw.shape[1]} cột")
st.divider()

# ==========================================
# KHỐI XỬ LÝ HUẤN LUYỆN (Chạy khi bấm nút)
# ==========================================
if train_button:
    with st.spinner("Đang tiền xử lý và huấn luyện mô hình..."):
        # 1. Tiền xử lý
        df = df_raw.copy()
        df = df.dropna(subset=FEATURES + [TARGET])
        
        # Mã hóa cột phân loại 'type'
        le = LabelEncoder()
        df['type'] = le.fit_transform(df['type'])
        
        X = df[FEATURES]
        y = df[TARGET]
        
        # 2. Chia tập
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        # 3. Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 4. Huấn luyện
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
        else:
            model = LogisticRegression(max_iter=max_iter, C=C_param, random_state=random_state)
            
        model.fit(X_train_scaled, y_train)
        
        # 5. Đánh giá nhanh
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
        # 6. Lưu vào session_state
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['label_encoder'] = le
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        st.session_state['y_proba'] = y_proba
        st.session_state['model_name'] = model_choice
        
    st.success("✅ Huấn luyện thành công! Xem chi tiết ở các Tab bên dưới.")

# ==========================================
# CÁC TAB HIỂN THỊ NỘI DUNG CHÍNH
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Tổng quan Dữ liệu", 
    "📈 Trực quan hóa Dữ liệu", 
    "⚙️ Kết quả Kiểm định", 
    "🚀 Sử dụng Mô hình"
])

# ------------------------------------------
# THÀNH PHẦN 3: TỔNG QUAN DỮ LIỆU
# ------------------------------------------
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Số lượng bản ghi", f"{df_raw.shape[0]:,}")
    col2.metric("Số lượng đặc trưng (cột)", f"{df_raw.shape[1]}")
    col3.metric("Tỷ lệ gian lận (%)", f"{(df_raw[TARGET].mean() * 100):.2f}%")
    
    st.subheader("Xem trước Dữ liệu")
    with st.container(height=300):
        st.dataframe(df_raw.head(100), use_container_width=True)
        
    st.subheader("Thống kê Mô tả (Biến đầu vào)")
    st.dataframe(df_raw[FEATURES].describe(), use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 4: TRỰC QUAN HÓA DỮ LIỆU
# ------------------------------------------
with tab2:
    st.subheader("Biểu đồ phân phối các biến quan trọng")
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        # Biến mục tiêu
        fig1 = px.pie(df_raw, names=TARGET, title="Tỷ lệ phân phối biến mục tiêu (isFraud)", hole=0.4, color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_v2:
        # Phân phối loại giao dịch
        type_counts = df_raw['type'].value_counts().reset_index()
        type_counts.columns = ['type', 'count']
        fig2 = px.bar(type_counts, x='type', y='count', title="Phân phối loại giao dịch (type)", color='type')
        st.plotly_chart(fig2, use_container_width=True)

    col_v3, col_v4 = st.columns(2)
    
    with col_v3:
        # Phân phối Amount (dùng log scale do chênh lệch lớn)
        fig3 = px.histogram(df_raw, x='amount', title="Phân phối Số tiền giao dịch (Amount - Log scale)", log_y=True, nbins=50)
        st.plotly_chart(fig3, use_container_width=True)
        
    with col_v4:
        # Boxplot amount vs fraud
        fig4 = px.box(df_raw, x=TARGET, y='amount', title="Số tiền giao dịch theo trạng thái Gian lận", log_y=True, color=TARGET)
        st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 5: KẾT QUẢ KIỂM ĐỊNH MÔ HÌNH
# ------------------------------------------
with tab3:
    if 'model' not in st.session_state:
        st.info("💡 Mô hình chưa được huấn luyện. Vui lòng thiết lập tham số ở thanh bên trái và bấm 'Huấn luyện Mô hình'.")
    else:
        st.subheader(f"Chỉ tiêu kiểm định: {st.session_state['model_name']}")
        
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        y_proba = st.session_state['y_proba']
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        c2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
        c3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
        c4.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Ma trận nhầm lẫn (Confusion Matrix)**")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', 
                               labels=dict(x="Dự báo", y="Thực tế"),
                               x=['Hợp lệ (0)', 'Gian lận (1)'], y=['Hợp lệ (0)', 'Gian lận (1)'])
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_m2:
            st.markdown("**Đường cong ROC**")
            if y_proba is not None:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc_score = roc_auc_score(y_test, y_proba)
                fig_roc = px.line(x=fpr, y=tpr, title=f"ROC Curve (AUC = {auc_score:.4f})", labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
                fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.warning("Mô hình không hỗ trợ xuất xác suất (predict_proba) để vẽ ROC.")

# ------------------------------------------
# THÀNH PHẦN 6: SỬ DỤNG MÔ HÌNH
# ------------------------------------------
with tab4:
    if 'model' not in st.session_state:
        st.info("💡 Vui lòng huấn luyện mô hình trước khi sử dụng để dự báo.")
    else:
        st.subheader("Dự báo giao dịch mới")
        mode = st.radio("Chọn phương thức nhập dữ liệu:", ["✏️ Nhập thủ công từng giao dịch", "📁 Tải file dự báo hàng loạt"])
        
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        le = st.session_state['label_encoder']
        
        if mode == "✏️ Nhập thủ công từng giao dịch":
            with st.form("predict_form"):
                st.markdown("**Nhập thông tin giao dịch:**")
                col_f1, col_f2 = st.columns(2)
                
                with col_f1:
                    f_step = st.number_input("Step (Giờ giao dịch)", value=1, min_value=1)
                    f_type = st.selectbox("Loại giao dịch (type)", options=le.classes_)
                    f_amount = st.number_input("Số tiền (amount)", value=1000.0, min_value=0.0)
                    f_oldOrg = st.number_input("Số dư gửi ban đầu (oldbalanceOrg)", value=0.0, min_value=0.0)
                
                with col_f2:
                    f_newOrg = st.number_input("Số dư gửi lúc sau (newbalanceOrig)", value=0.0, min_value=0.0)
                    f_oldDest = st.number_input("Số dư nhận ban đầu (oldbalanceDest)", value=0.0, min_value=0.0)
                    f_newDest = st.number_input("Số dư nhận lúc sau (newbalanceDest)", value=0.0, min_value=0.0)
                
                submit_pred = st.form_submit_button("🔍 Chẩn đoán Giao dịch")
                
                if submit_pred:
                    # Chuẩn bị dữ liệu
                    input_df = pd.DataFrame([{
                        'step': f_step,
                        'type': le.transform([f_type])[0],
                        'amount': f_amount,
                        'oldbalanceOrg': f_oldOrg,
                        'newbalanceOrig': f_newOrg,
                        'oldbalanceDest': f_oldDest,
                        'newbalanceDest': f_newDest
                    }])
                    
                    # Transform và Predict
                    input_scaled = scaler.transform(input_df)
                    pred = model.predict(input_scaled)[0]
                    prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None
                    
                    st.divider()
                    if pred == 1:
                        st.error(f"🚨 **CẢNH BÁO:** Đây có thể là GIAO DỊCH GIAN LẬN! (Xác suất: {prob*100:.2f}%)")
                    else:
                        st.success(f"✅ Giao dịch hợp lệ. (Xác suất gian lận: {prob*100:.2f}%)")
                        
        else:
            upload_test = st.file_uploader("Tải file CSV/Excel cần dự báo", type=['csv', 'xlsx'], key="upload_test")
            if upload_test is not None:
                df_test_raw = load_data(upload_test, upload_test.name)
                
                # Kiểm tra cột
                missing = [c for c in FEATURES if c not in df_test_raw.columns]
                if missing:
                    st.error(f"File thiếu các cột: {', '.join(missing)}")
                else:
                    st.success("File hợp lệ! Bấm nút dưới đây để dự báo.")
                    if st.button("Dự báo Hàng loạt"):
                        df_pred = df_test_raw.copy()
                        # Xử lý các nhãn type có thể chưa xuất hiện lúc train
                        known_classes = list(le.classes_)
                        df_pred['type_encoded'] = df_pred['type'].apply(lambda x: le.transform([x])[0] if x in known_classes else -1)
                        # Nếu có nhãn chưa biết, có thể cảnh báo. Để đơn giản, giả sử data khớp.
                        
                        X_test_batch = df_pred[FEATURES].copy()
                        X_test_batch['type'] = df_pred['type_encoded']
                        
                        X_test_batch_scaled = scaler.transform(X_test_batch)
                        preds = model.predict(X_test_batch_scaled)
                        
                        df_test_raw['Prediction_isFraud'] = preds
                        
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(X_test_batch_scaled)[:, 1]
                            df_test_raw['Probability_Fraud'] = np.round(probs, 4)
                            
                        with st.container(height=300):
                            st.dataframe(df_test_raw)
                            
                        csv = df_test_raw.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 Tải Bảng Kết quả (CSV)",
                            data=csv,
                            file_name="predicted_transactions.csv",
                            mime="text/csv"
                        )
