import streamlit as st
import requests
from typing import Dict

# Конфигурация
BACKEND_URL = "http://localhost:8000"  # URL вашего FastAPI сервера
MODELS = ["simple_nn", "distilbert"]
LABELS = ["спорт", "юмор", "реклама", "соцсети", "политика", "личная жизнь"]

def main():
    st.title("Классификация текста по темам")
    st.markdown("""
    Определение тематики текста по следующим категориям:
    - Спорт
    - Юмор
    - Реклама
    - Соцсети
    - Политика
    - Личная жизнь
    """)

    show_model_ratings()

    text = st.text_area("Введите текст для анализа:", height=150)
    uploaded_file = st.file_uploader("Или загрузите текстовый файл:", type=["txt"])
    
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
    
    if st.button("Анализировать текст"):
        if not text:
            st.error("Пожалуйста, введите текст или загрузите файл")
            return

        with st.spinner("Анализируем текст..."):
            if uploaded_file:
                response = predict_file(uploaded_file)
            else:
                response = predict_text(text)
            
            if response:
                display_results(response)
                rating_section()

def predict_text(text: str) -> Dict:
    """Отправка текста на бэкенд для анализа"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict_text",
            json={"text": text}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка подключения к серверу: {e}")
        return None

def predict_file(file) -> Dict:
    """Отправка файла на бэкенд для анализа"""
    try:
        files = {"file": (file.name, file.getvalue(), "text/plain")}
        response = requests.post(
            f"{BACKEND_URL}/predict_file",
            files=files
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка подключения к серверу: {e}")
        return None

def display_results(results: Dict):
    """Отображение результатов классификации"""
    st.subheader("Результаты анализа:")
    
    for model_name in MODELS:
        st.markdown(f"### Модель: {model_name.upper()}")
        model_results = results.get(model_name, {})
        
        cols = st.columns(2)
        for i, (label, score) in enumerate(model_results.items()):
            with cols[i % 2]:
                st.progress(
                    value=float(score),
                    text=f"{label.capitalize()}: {score:.2%}"
                )

def rating_section():
    """Секция для оценки моделей"""
    st.subheader("Оцените качество работы моделей:")
    
    with st.form("rating_form"):
        cols = st.columns(2)
        ratings = {}
        
        for i, model in enumerate(MODELS):
            with cols[i % 2]:
                ratings[model] = st.slider(
                    f"Оценка для {model}",
                    1, 5, 3,
                    key=f"rating_{model}"
                )
        
        if st.form_submit_button("Отправить оценки"):
            submit_ratings(ratings)

def submit_ratings(ratings: Dict):
    """Отправка оценок на бэкенд"""
    try:
        for model, rating in ratings.items():
            response = requests.post(
                f"{BACKEND_URL}/rate_model",
                json={
                    "model_name": model,
                    "rating": rating
                }
            )
            response.raise_for_status()
        
        st.success("Спасибо за вашу оценку!")
        st.experimental_rerun()
        
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка отправки оценки: {e}")

def show_model_ratings():
    """Показать текущие рейтинги моделей"""
    try:
        response = requests.get(f"{BACKEND_URL}/model_ratings")
        response.raise_for_status()
        ratings = response.json()
        
        st.sidebar.subheader("Текущие рейтинги моделей:")
        for model, score in ratings.items():
            st.sidebar.markdown(
                f"**{model.upper()}**: {score:.1f} ★" if score else 
                f"**{model.upper()}**: Нет оценок"
            )
    
    except requests.exceptions.RequestException:
        st.sidebar.error("Не удалось загрузить рейтинги моделей")

if __name__ == "__main__":
    main()