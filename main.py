import streamlit as st
import requests
import json
import time

API_BASE = "http://localhost:8000/users"

st.set_page_config(
    page_title="PCPAi AI Platform",
    page_icon="assets/OK_upscayl_1x_realesrgan-x4plus.png",
    layout="wide",
)

st.markdown(
    """
    <style>
        @font-face {
            font-family: 'IRANSansWeb';
            src: url('IRANSansWeb.woff2') format('woff2');
        }

        html, body, [class*="css"] {
            font-family: 'IRANSansWeb', sans-serif !important;
        }
        h4{
            color: black !important;
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
        }

        @keyframes gradientAnimation {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        [data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(135deg, #6c63ff, #ff6ec4, #00c9ff, #6c63ff);
            background-size: 400% 400%;
            animation: gradientAnimation 15s ease infinite;
            padding: 1.5rem;
            color: white;
            border-top-right-radius: 12px;
            border-bottom-right-radius: 12px;
        }

        .sidebar .block-container img {
            max-width: 90px !important;
            margin: 10px auto;
            display: block;
        }

        .stRadio > div {
            font-size: 1.2rem;
            font-weight: bold;
            color: white;
        }

        .stRadio div[role="radiogroup"] label {
            padding: 0.6rem 1.2rem;
            border-radius: 0.5rem;
            transition: background-color 0.3s ease;
        }

        .stRadio div[role="radiogroup"] label:hover {
            background-color: rgba(255, 255, 255, 0.2);
            cursor: pointer;
        }

        .main {
            background: linear-gradient(to bottom right, #f0f0ff, #ffffff);
            font-family: 'IRANSansWeb', sans-serif;
            padding: 2rem;
        }

        .stContainer {
            border-radius: 1rem;
            background-color: white;
            padding: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }

        button[kind="primary"] {
            background: linear-gradient(135deg, #6c63ff, #00c9ff);
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
        }

        button[kind="primary"]:hover {
            background: linear-gradient(135deg, #5a52e0, #00a0d6);
            transform: scale(1.03);
        }

        h1, h2, h3 {
            color: #4b4b7e;
        }

        pre {
            background-color: #f6f6ff;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .alert {
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            font-weight: bold;
            animation: fadeInSlide 0.8s ease-out forwards;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .alert-success {
            background-color: #d4edda;
            color: #155724;
            border-left: 5px solid #28a745;
        }

        .alert-error {
            background-color: #f8d7da;
            color: #721c24;
            border-left: 5px solid #dc3545;
        }
        .alert-warning {
            background-color: #fff3cd;
            color: #856404;
            border-left: 5px solid #ffc107;
        }

        @keyframes fadeInSlide {
            0% {
                opacity: 0;
                transform: translateY(-10px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .article-btn {
            background: linear-gradient(135deg, #6c63ff, #00c9ff);
            border: none;
            color: white;
            padding: 8px 16px;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
            width: 100%;
            text-align: center;
            user-select: none;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }

        .article-btn:hover {
            background: linear-gradient(135deg, #5a52e0, #00a0d6);
            transform: translateY(-4px) scale(1.05);
            box-shadow: 0 10px 20px rgba(0,0,0,0.25);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "token" not in st.session_state:
    st.session_state.token = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None

headers = {}
if st.session_state.token:
    headers = {"Authorization": f"Bearer {st.session_state.token}"}

with st.sidebar:
    st.image("assets/OK_upscayl_1x_realesrgan-x4plus.png", use_container_width=True)
    st.markdown("<h4>PCPAi AI Platform</h4>", unsafe_allow_html=True)
    menu = (
        ["Login", "Register"] if not st.session_state.token else ["Dashboard", "Logout"]
    )
    choice = st.radio("📂 Menu", menu)


def show_alert(message, type="success"):
    icons = {"success": "✅", "error": "❌", "warning": "⚠️"}
    css_classes = {
        "success": "alert-success",
        "error": "alert-error",
        "warning": "alert-warning",
    }
    icon = icons.get(type, "ℹ️")
    css_class = css_classes.get(type, "alert-warning")
    st.markdown(
        f'<div class="alert {css_class}">{icon} {message}</div>',
        unsafe_allow_html=True,
    )


def login():
    st.title("🔐 Login")
    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")
    if st.button("✅ Login"):
        if not username or not password:
            show_alert("لطفاً تمام فیلدها را پر کنید.", type="warning")
            return
        res = requests.post(
            f"{API_BASE}/login", json={"username": username, "password": password}
        )
        if res.status_code == 200:
            token_data = res.json()[0]
            st.session_state.token = token_data["token"]
            st.session_state.user_id = token_data["user_id"]
            st.session_state.username = username
            show_alert("✅ Logged in successfully", type="success")
            time.sleep(2000)
            st.rerun()
        else:
            show_alert(res.json()["detail"], type="error")


def register():
    st.title("📝 Create New Account")
    username = st.text_input("👤 Username")
    phone_number = st.text_input("📞 Phone Number")
    degree = st.text_input("🎓 Field/Degree")
    password = st.text_input("🔒 Password", type="password")
    confirm_password = st.text_input("🔁 Confirm Password", type="password")
    if st.button("📥 Register"):
        if (
            not username
            or not phone_number
            or not degree
            or not password
            or not confirm_password
        ):
            show_alert("لطفاً تمام فیلدها را پر کنید.", type="warning")
            return
        if password != confirm_password:
            show_alert("رمز عبور و تکرار آن یکسان نیستند.", type="error")
            return
        data = {
            "username": username,
            "phone_number": phone_number,
            "degree": degree,
            "password": password,
            "confirm_password": confirm_password,
        }
        res = requests.post(f"{API_BASE}/register", json=data)
        if res.status_code == 200:
            show_alert("✅ Registered successfully. Logging in...", type="success")
            login_res = requests.post(
                f"{API_BASE}/login", json={"username": username, "password": password}
            )
            if login_res.status_code == 200:
                token_data = login_res.json()[0]
                st.session_state.token = token_data["token"]
                st.session_state.user_id = token_data["user_id"]
                st.session_state.username = username
                st.rerun()
        else:
            show_alert(res.json()["detail"], type="error")


def dashboard():
    st.title(f"📊 Dashboard | Welcome {st.session_state.username}")

    st.subheader("📝 Add New Text")
    new_text = st.text_area("📄 Enter your text here")
    if st.button("💾 Save Text"):
        res = requests.post(
            f"{API_BASE}/texts/",
            params={"user_id": st.session_state.user_id},
            json={"text_content": new_text},
        )
        if res.status_code == 200:
            show_alert("✅ Text saved successfully", type="success")
            st.rerun()
        else:
            show_alert(res.json()["detail"], type="error")

    st.subheader("📜 My Texts")
    texts_res = requests.get(f"{API_BASE}/texts/{st.session_state.user_id}")
    if texts_res.status_code == 200:
        texts = texts_res.json()
        selected_text = st.selectbox(
            "✏️ Select a text", texts, format_func=lambda x: x["content"][:50]
        )
        if st.button("🗑️ Delete this text"):
            del_res = requests.delete(f"{API_BASE}/texts/{selected_text['id']}")
            if del_res.status_code == 200:
                show_alert("🗑️ Text deleted successfully", type="success")
                st.rerun()
            else:
                show_alert("❌ Error deleting text", type="error")

        header = st.text_input("📰 Article Title")
        if st.button("🚀 Generate Article"):
            res = requests.post(
                f"{API_BASE}/articles/process",
                params={
                    "user_id": st.session_state.user_id,
                    "text_id": selected_text["id"],
                },
                data=json.dumps(header),
                headers={"Content-Type": "application/json"},
            )
            if res.status_code == 200:
                data = res.json()
                if "generated_article" in data:
                    show_alert("✅ Article generated", type="success")
                    st.json(data["generated_article"])
                else:
                    show_alert(
                        "⚠️ No article generated. Try a more scientific text or specific title.",
                        type="warning",
                    )
            else:
                detail = res.json().get("detail", "")
                if "No articles found based on the keywords" in detail:
                    show_alert(
                        "🔍 No related article found. Try more specialized content or keywords.",
                        type="warning",
                    )
                else:
                    show_alert("❌ Error generating article: " + detail, type="error")

        st.subheader("📚 Generated Articles")
        article_res = requests.get(f"{API_BASE}/articles/{st.session_state.user_id}")
        if article_res.status_code == 200:
            for a in article_res.json():
                st.markdown(f"#### 📝 {a['title']}")
                st.write(a["model_output"])
                st.write(f"🕒 Generated on: {a['created_at']}")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button(f"🌐 Translate - {a['id']}"):
                        trans_res = requests.post(
                            f"{API_BASE}/articles/{a['id']}/translate"
                        )
                        st.success(
                            trans_res.json().get(
                                "translated_text", "❌ Translation failed"
                            )
                        )

                with col2:
                    if st.button(f"📄 PDF - {a['id']}"):
                        pdf_res = requests.post(
                            f"{API_BASE}/articles/{a['id']}/generate_pdf",
                            params={"user_id": st.session_state.user_id},
                        )
                        if pdf_res.status_code == 200:
                            st.markdown(
                                f"[📥 Download PDF]({pdf_res.json()['download_url']})"
                            )

                with col3:
                    if st.button(f"📝 Word - {a['id']}"):
                        word_res = requests.post(
                            f"{API_BASE}/articles/{a['id']}/generate_word",
                            params={"user_id": st.session_state.user_id},
                        )
                        if word_res.status_code == 200:
                            st.markdown(
                                f"[📥 Download Word]({word_res.json()['download_url']})"
                            )

                with col4:
                    if st.button(f"🗑️ Delete - {a['id']}"):
                        del_res = requests.delete(f"{API_BASE}/articles/{a['id']}")
                        if del_res.status_code == 200:
                            show_alert("🗑️ Article deleted", type="success")
                            st.rerun()
                        else:
                            show_alert("❌ Error deleting article", type="error")
        else:
            st.info("⛔ No articles available.")
    else:
        show_alert("⚠️ Error retrieving texts.", type="warning")


def main():
    if choice == "Login":
        login()
    elif choice == "Register":
        register()
    elif choice == "Dashboard":
        dashboard()
    elif choice == "Logout":
        st.session_state.token = None
        st.session_state.username = None
        st.session_state.user_id = None
        st.rerun()


if __name__ == "__main__":
    main()
