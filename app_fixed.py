import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ==========================================
# PHẦN 1: TOÁN HỌC & KIỂM TRA
# ==========================================

def check_system(A):
    """Kiểm tra các điều kiện của ma trận trước khi giải"""
    det_A = np.linalg.det(A)
    if np.abs(det_A) < 1e-12:
        return False, "🔴 CẢNH BÁO MỨC ĐỎ: Định thức ma trận bằng 0 (hoặc cực kỳ nhỏ). Hệ vô nghiệm hoặc vô số nghiệm."
    
    # Kiểm tra tính chéo trội
    D = np.abs(np.diag(A))
    S = np.sum(np.abs(A), axis=1) - D
    is_sdd = np.all(D > S)
    
    msg = "✅ Ma trận an toàn để giải." if is_sdd else "⚠️ LƯU Ý: Ma trận KHÔNG chéo trội. Phương pháp lặp (Jacobi/Gauss-Seidel) có nguy cơ phân kỳ!"
    return True, msg

def solve_lu_pivoting(A, b):
    """Phân rã LU với Xoay trục bán phần (Chống sai số làm tròn)"""
    n = len(A)
    U = A.astype(float).copy()
    L = np.eye(n, dtype=float)
    P = np.eye(n, dtype=float)
    b_new = b.astype(float).copy()

    for i in range(n):
        max_idx = np.argmax(np.abs(U[i:n, i])) + i
        if max_idx != i:
            U[[i, max_idx]] = U[[max_idx, i]]
            P[[i, max_idx]] = P[[max_idx, i]]
            b_new[[i, max_idx]] = b_new[[max_idx, i]]
            if i > 0:
                L[[i, max_idx], :i] = L[[max_idx, i], :i]

        if np.abs(U[i, i]) < 1e-12:
            raise ValueError("Phần tử chéo bằng 0, không thể tiếp tục khử.")

        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]

    y = np.zeros_like(b_new)
    for i in range(n):
        y[i] = b_new[i] - np.dot(L[i, :i], y[:i])

    x = np.zeros_like(b_new)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x, L, U, P

def solve_jacobi_optimized(A, b, eps, max_iter=100):
    """Phương pháp Jacobi tối ưu hóa"""
    n = len(A)
    x = np.zeros_like(b, dtype=float)
    errors = []
    iterations_data = []
    
    D = np.diag(A).reshape(-1, 1)
    if np.any(np.abs(D) < 1e-12):
        raise ValueError("Lỗi: Có phần tử trên đường chéo bằng 0, phương pháp Jacobi thất bại.")
        
    R = A - np.diagflat(np.diag(A))
    
    for k in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        error = np.linalg.norm(x_new - x, ord=np.inf)
        errors.append(error)
        
        iterations_data.append({
            'Lần lặp': k + 1,
            'Sai số': error,
            'Nghiệm': x_new.flatten().tolist()
        })
        
        x = x_new
        
        if error < eps:
            return x, k+1, errors, iterations_data
            
    return x, max_iter, errors, iterations_data

def solve_gauss_seidel_strict(A, b, eps, max_iter=100):
    """Gauss-Seidel với kiểm định Định lý 3.4"""
    n = len(A)
    x = np.zeros_like(b, dtype=float)
    errors = []
    iterations_data = []
    
    A1 = np.tril(A)
    A2 = A - A1
    
    A1_inv = np.linalg.inv(A1)
    T = -np.dot(A1_inv, A2)
    
    lambda_val = np.linalg.norm(T, ord=np.inf)
    
    if lambda_val >= 1:
        return None, 0, [], [], lambda_val
    
    error_multiplier = lambda_val / (1.0 - lambda_val)
    
    for k in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            sum_val = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - sum_val) / A[i, i]
            
        step_diff = np.linalg.norm(x - x_old, ord=np.inf)
        true_error_bound = error_multiplier * step_diff
        errors.append(true_error_bound)
        
        iterations_data.append({
            'Lần lặp': k + 1,
            'Sai số': true_error_bound,
            'Khoảng cách 2 bước': step_diff,
            'Nghiệm': x.flatten().tolist()
        })
        
        if true_error_bound <= eps:
            return x, k+1, errors, iterations_data, lambda_val
            
    return x, max_iter, errors, iterations_data, lambda_val


# ==========================================
# PHẦN 2: GIAO DIỆN STREAMLIT
# ==========================================

st.set_page_config(
    page_title="Hệ Phương Trình Tuyến Tính",
    page_icon="🔢",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    /* Khử khoảng trắng thừa khi label bị ẩn */
    div[data-testid="stNumberInput"] > label[data-visibility="hidden"] {
        display: none !important;
    }
    div[data-testid="stNumberInput"] {
        margin-bottom: 4px !important;
    }
    .method-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .error-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🔢 PHÂN TÍCH VÀ GIẢI HỆ PHƯƠNG TRÌNH TUYẾN TÍNH AX = B</h1>
    <p>Phương pháp LU, Jacobi và Gauss-Seidel</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📊 NHẬP DỮ LIỆU")
    
    input_method = st.radio(
        "Chọn phương thức nhập:",
        ["📝 Nhập thủ công", "📁 Tải file"]
    )
    
    if input_method == "📝 Nhập thủ công":
        n = st.number_input("Kích thước ma trận n:", min_value=2, max_value=10, value=3)
        
        st.markdown("**Ma trận A**")
        st.markdown("<p style='margin:0 0 6px 0; font-size:13px; color:#aaa; font-style:italic;'>Nhập các hệ số của ma trận A</p>", unsafe_allow_html=True)
        A_data = []
        for i in range(n):
            cols = st.columns(n)
            row = []
            for j in range(n):
                row.append(cols[j].number_input(f"a{i+1}{j+1}", value=0.0, key=f"A_{i}_{j}", label_visibility="hidden"))
            A_data.append(row)
        A = np.array(A_data)

        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        st.markdown("**Vector b**")
        st.markdown("<p style='margin:0 0 6px 0; font-size:13px; color:#aaa; font-style:italic;'>Nhập vế phải của hệ phương trình</p>", unsafe_allow_html=True)
        b_data = []
        cols = st.columns(n)
        for i in range(n):
            b_data.append(cols[i].number_input(f"b{i+1}", value=0.0, key=f"b_{i}", label_visibility="hidden"))
        b = np.array(b_data).reshape(-1, 1)
        
    else:
        uploaded_file = st.file_uploader("Chọn file (.csv hoặc .txt)", type=['csv', 'txt'])
        if uploaded_file is not None:
            try:
                delimiter = ',' if uploaded_file.name.endswith('.csv') else None
                data = np.loadtxt(uploaded_file, delimiter=delimiter)
                A = data[:, :-1]
                b = data[:, -1:]
                n = len(A)
                st.success(f"✅ Đã đọc dữ liệu từ {uploaded_file.name}")
            except Exception as e:
                st.error(f"Lỗi đọc file: {e}")
                A = None
                b = None
        else:
            A = None
            b = None
    
    eps = st.number_input(
        "Sai số epsilon:", 
        value=1e-6, 
        format="%.1e",
        help="Sai số cho phương pháp lặp"
    )

if A is not None and b is not None:
    is_valid, message = check_system(A)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if is_valid:
            st.markdown(f'<div class="success-box">{message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="error-box">{message}</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Kích thước hệ", f"{len(A)}x{len(A)}")
        st.metric("Định thức", f"{np.linalg.det(A):.6f}")
    
    st.subheader("📋 Ma trận đã nhập")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Ma trận A:**")
        st.dataframe(pd.DataFrame(A, columns=[f"x{i+1}" for i in range(len(A))]))
    with col_b:
        st.write("**Vector b:**")
        st.dataframe(pd.DataFrame(b, columns=["b"]))
    
    st.subheader("⚙️ CHỌN PHƯƠNG PHÁP GIẢI")
    
    method = st.selectbox(
        "Phương pháp:",
        ["🔷 LU với xoay trục", "🔶 Jacobi", "🔰 Gauss-Seidel"]
    )
    
    if st.button("🚀 GIẢI HỆ PHƯƠNG TRÌNH", type="primary", use_container_width=True):
        with st.spinner("Đang tính toán..."):
            if "LU" in method:
                try:
                    x, L, U, P = solve_lu_pivoting(A, b)
                    
                    st.markdown("### 📊 KẾT QUẢ PHÂN RÃ LU")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**Ma trận P:**")
                        st.dataframe(pd.DataFrame(P))
                    with col2:
                        st.write("**Ma trận L:**")
                        st.dataframe(pd.DataFrame(np.round(L, 4)))
                    with col3:
                        st.write("**Ma trận U:**")
                        st.dataframe(pd.DataFrame(np.round(U, 4)))
                    
                    st.markdown("### ✅ VECTOR NGHIỆM")
                    solution_df = pd.DataFrame({
                        'Biến': [f'x{i+1}' for i in range(len(x))],
                        'Giá trị': x.flatten()
                    })
                    st.dataframe(solution_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Lỗi: {e}")
                    
            elif "Jacobi" in method:
                try:
                    x, iters, errors, iter_data = solve_jacobi_optimized(A, b, eps)
                    
                    st.markdown(f"### ✅ KẾT QUẢ JACOBI (Hội tụ sau {iters} bước)")
                    
                    solution_df = pd.DataFrame({
                        'Biến': [f'x{i+1}' for i in range(len(x))],
                        'Giá trị': x.flatten()
                    })
                    st.dataframe(solution_df, use_container_width=True)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(errors)+1)),
                        y=errors,
                        mode='lines+markers',
                        name='Sai số',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=6)
                    ))
                    
                    fig.update_layout(
                        title="📈 Đồ thị hội tụ - Phương pháp Jacobi",
                        xaxis_title="Số bước lặp (k)",
                        yaxis_title="Sai số (thang log)",
                        yaxis_type="log",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("📊 Xem chi tiết các bước lặp"):
                        df_iter = pd.DataFrame(iter_data)
                        st.dataframe(df_iter, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Lỗi: {e}")
                    
            else:  # Gauss-Seidel
                try:
                    result = solve_gauss_seidel_strict(A, b, eps)
                    
                    if result[0] is None:
                        lambda_val = result[4]
                        st.markdown(f"""
                        <div class="error-box">
                            <h3>🔴 THẤT BẠI - PHƯƠNG PHÁP PHÂN KỲ</h3>
                            <p>Hệ số Lambda = {lambda_val:.6f} ≥ 1</p>
                            <p>Thuật toán đã bị chặn để bảo vệ hệ thống. Vui lòng sử dụng ma trận khác thỏa mãn tính chéo trội!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        x, iters, errors, iter_data, lambda_val = result
                        
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>✅ KIỂM ĐỊNH THÀNH CÔNG</h3>
                            <p>Hệ số Lambda = {lambda_val:.6f} < 1</p>
                            <p>Hệ số nhân sai số = {lambda_val/(1-lambda_val):.6f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"### ✅ KẾT QUẢ GAUSS-SEIDEL (Hội tụ sau {iters} bước)")
                        
                        solution_df = pd.DataFrame({
                            'Biến': [f'x{i+1}' for i in range(len(x))],
                            'Giá trị': x.flatten()
                        })
                        st.dataframe(solution_df, use_container_width=True)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(errors)+1)),
                            y=errors,
                            mode='lines+markers',
                            name='Sai số ước lượng',
                            line=dict(color='#2ca02c', width=2),
                            marker=dict(size=6)
                        ))
                        
                        if iter_data and 'Khoảng cách 2 bước' in iter_data[0]:
                            step_diffs = [d['Khoảng cách 2 bước'] for d in iter_data]
                            fig.add_trace(go.Scatter(
                                x=list(range(1, len(step_diffs)+1)),
                                y=step_diffs,
                                mode='lines+markers',
                                name='Khoảng cách 2 bước',
                                line=dict(color='#ff7f0e', width=2, dash='dash'),
                                marker=dict(size=4)
                            ))
                        
                        fig.update_layout(
                            title="📈 Đồ thị hội tụ - Phương pháp Gauss-Seidel",
                            xaxis_title="Số bước lặp (k)",
                            yaxis_title="Sai số (thang log)",
                            yaxis_type="log",
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        with st.expander("📊 Xem chi tiết các bước lặp"):
                            df_iter = pd.DataFrame(iter_data)
                            st.dataframe(df_iter, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Lỗi: {e}")

else:
    st.info("👈 Vui lòng nhập dữ liệu ở sidebar để bắt đầu!")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🔬 Chương trình giải hệ phương trình tuyến tính - Phương pháp số</p>
</div>
""", unsafe_allow_html=True)
