# Dark@Web FPCP – AI-Powered Scientific Article Generation Platform

<p align="center">
  <img src="https://img.shields.io/badge/Status-Production Ready-success?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Llama--3--70B--4bit-FF6F61?style=for-the-badge" alt="Llama-3-70B">
  <img src="https://img.shields.io/badge/FAISS%20%2B%20BM25-Hybrid%20Search-28a745?style=for-the-badge" alt="FAISS + BM25">
  <img src="https://img.shields.io/badge/Multi--Agent%20RAG-Advanced-00ffff?style=for-the-badge" alt="Multi-Agent">
</p>

<p align="center">
  <span style="font-size: 3.8em;">🧠</span><br>
  <strong style="font-size: 2.5em; background: linear-gradient(90deg, #00ffff, #0080ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
    DARK@WEB FPCP
  </strong><br>
  <i>پلتفرم پیشرفته تولید خودکار مقالات علمی با هوش مصنوعی</i>
</p>

## 🚀 معرفی پروژه

**FPCP AI** یک پلتفرم کاملاً خودکار و حرفه‌ای برای تولید مقالات علمی، پایان‌نامه، ژورنال پیپر و محتوای آکادمیک با استفاده از هوش مصنوعی نسل جدید است.

این سیستم با بهره‌گیری از معماری چندعاملی (Multi-Agent)، Retrieval-Augmented Generation (RAG)، مدل 70 میلیاردی Llama-3 با بهینه‌سازی LoRA و جستجوی ترکیبی FAISS + BM25، قادر است در کمتر از 90 ثانیه یک مقاله علمی کامل، اصیل، قابل استناد و با کیفیت ژورنال تولید کند.

مناسب برای دانشجویان دکتری، اساتید، پژوهشگران و ناشران علمی.

## ✨ قابلیت‌های کلیدی

| قابلیت                            | توضیحات                                                                                 |
|-----------------------------------|----------------------------------------------------------------------------------------|
| Multi-Agent Architecture          | Planner → Retriever → Generator → Critic → Post-Processor                             |
| Hybrid Retrieval                  | جستجوی معنایی (FAISS L2) + کلیدواژه‌ای (BM25) + CrossRef API                          |
| Llama-3-70B 4bit + LoRA           | تولید متن فوق‌العاده باکیفیت و تنظیم‌شده روی داده‌های PubMed                         |
| تطبیق خودکار با سطح کاربر        | تولید متن ساده (کارشناسی) تا پیشرفته (دکتری/استاد)                                   |
| ارزیابی خودکار کیفیت              | Critic Agent با DeBERTa + مدل اختصاصی Coherence                                        |
| خروجی چندفرمتی                    | PDF • Word • LaTeX • Markdown                                                        |
| ترجمه خودکار به فارسی و انگلیسی   | ترجمه متون طولانی با روش Chunked (بدون محدودیت کاراکتر)                                |
| تولید شکل و نمودار خودکار         | با Matplotlib و امکان درج در PDF/Word/LaTeX                                           |
| کنترل سرقت ادبی                   | تولید محتوای 100% اصیل با حداکثر شباهت زیر 12% به منابع                                |
| پایگاه دانش داخلی + خارجی          | ذخیره‌سازی دائمی مقالات + دریافت لحظه‌ای از CrossRef                                |


## 🔧 فناوری‌های به‌کاررفته

- **Backend**: FastAPI + SQLAlchemy 2.0 (Async)
- **دیتابیس**: PostgreSQL
- **مدل اصلی**: Llama-3-70B-Instruct (4-bit quantized) + LoRA fine-tuning روی PubMed
- **Embedding**: sentence-transformers/all-mpnet-base-v2
- **Vector Search**: FAISS (L2 normalized) + BM25
- **منبع خارجی**: CrossRef API
- **Critic Model**: microsoft/deberta-v3-large + Custom Coherence Head
- **ترجمه**: Google Translate API (chunked)
- **تولید فایل**: FPDF • python-docx • pylatexenc • Matplotlib
- **احراز هویت**: JWT + bcrypt

## 📊 عملکرد (آزمون داخلی روی 100 مقاله)

| معیار                     | مقدار               |
|---------------------------|---------------------|
| میانگین ROUGE-L           | 0.72                |
| میانگین Coherence Score  | 0.91                |
| حداکثر شباهت به منبع     | ≤ 12%               |
| زمان تولید مقاله ~4000 کلمه | 45–90 ثانیه        |

## 🚀 راه‌اندازی سریع (Development)

```bash
git clone https://github.com/DARK000WEB/FPCP_AI.git
cd FPCP_AI

python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

pip install -r requirements.txt

cp .env.example .env
.

alembic upgrade head

uvicorn main:app --reload --port=8000