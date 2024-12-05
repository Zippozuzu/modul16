import hashlib
import streamlit as st
import sqlite3
from streamlit_option_menu import option_menu
from PIL import Image
from io import BytesIO
import tensorflow as tf
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

my_model = load_model('gs://goit_models/my_model_tf.keras')
my_model_vgg = load_model('gs://goit_models/vgg16_tf.keras')
# Створення або підключення до SQLite бази даних
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Створення таблиці, якщо вона не існує
c.execute('''
CREATE TABLE IF NOT EXISTS users(
    username TEXT PRIMARY KEY,
    password TEXT)
''')
conn.commit()

# Ініціалізація сесії
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''

# Функції для хешування паролів
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Функції для перевірки паролів
def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# Функції для додавання користувачів
def add_userdata(username, password):
    c.execute('INSERT INTO users(username, password) VALUES (?, ?)', (username, password))
    conn.commit()

# Функції для входу
def login_user(username, password):
    c.execute('SELECT * FROM users WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    return data

# Функції для виходу
def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = ''


def plot_predictions(predictions, class_names, title="Ймовірності для кожного класу", color="skyblue"):
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_probs = predictions[sorted_indices]
    sorted_classes = [class_names[i] for i in sorted_indices]

    # Побудова графіка
    plt.figure(figsize=(8, 6))
    plt.barh(sorted_classes, sorted_probs, color=color)
    plt.xlabel("Ймовірність", fontsize=12)
    plt.ylabel("Клас", fontsize=12)
    plt.title(title, fontsize=14)
    plt.gca().invert_yaxis()  

    return plt


# Навігація за допомогою option_menu
selected = option_menu(menu_title=None, options=["Home", "SignUp", "Login", "Profile", "Models"],
                       icons=["house", "person-plus", "log-in", "person", "gear"], menu_icon="cast", default_index=0,
                       orientation="horizontal")

if selected == "Home":
    st.title("Home Page")
    st.write("Welcome to the application!")

elif selected == "SignUp":
    st.title("Sign Up")
    new_user = st.text_input("Choose a username")
    new_password = st.text_input("Choose a password", type='password')
    if st.button("Sign Up"):
        add_userdata(new_user, make_hashes(new_password))
        st.success("You have successfully signed up!")

elif selected == "Login":
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        result = login_user(username, make_hashes(password))
        if check_hashes(password, result[0][1]):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome back, {username}!")
        else:
            st.error("Incorrect username or password.")

elif selected == "Profile":
    if st.session_state.logged_in:
        st.title("Your Profile")
        st.write(f"Username: {st.session_state.username}")
        if st.button("Logout"):
            logout()
    else:
        st.error("You are not logged in. Please login to see this page.")


elif selected == "Models":
    if st.session_state.logged_in:
        st.title("My models")
        model_type = st.sidebar.radio('Виберіть модель:', ['Моя згорткова нейромережа', 'VGG16'])
        # Заголовок сторінки
        st.title("Завантажте файт")
        
        uploaded_file = st.file_uploader("Виберіть файл...", type=["png","jpeg", ])

        if uploaded_file is not None:
            try:
                # Открываем загруженный файл
                img = Image.open(uploaded_file)

                grayscale_img = img.convert("L")

                resized_img = grayscale_img.resize((28, 28))

                # Сохраняем измененное изображение в буфер
                buffer = BytesIO()
                resized_img.save(buffer, format="PNG")
                buffer.seek(0)

                # Показываем измененное изображение
                st.image(resized_img, caption="Picture in 28x28", use_container_width=False)

            except Exception as e:
                st.error(f"Помилка обробки зображення: {e}")

        run_button = st.button("Run")
        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        if run_button:
            if model_type == "Моя згорткова нейромережа":
                try:
                    img_array = np.array(resized_img, dtype=np.float32)  
                    image_to_model = img_array.reshape((1, 28, 28, 1)) / 255.0  
                    
                    predictions = my_model.predict(image_to_model)

                    plt = plot_predictions(predictions[0], class_names, title="Ймовірності класів", color="skyblue")
                    st.pyplot(plt)  # Відображення графіка у Streamlit
                    predicted_label = class_names[np.argmax(predictions)]
                    st.write(f'Передбачення: {predicted_label}')

                except Exception as e:
                    st.error(f"Помилка обробки зображення: {e}")

            elif model_type == "VGG16":
                try:
   
                    img_array = np.array(grayscale_img, dtype=np.float32)  
                    img_tensor = tf.expand_dims(img_array, axis=-1)  
                    img_tensor_rgb = tf.image.grayscale_to_rgb(img_tensor)  
                    image_to_model = tf.image.resize_with_pad(img_tensor_rgb, 32, 32) / 255.0  
                    

                    image_to_model = tf.expand_dims(image_to_model, axis=0)


                    predictions = my_model_vgg.predict(image_to_model)

                    plt = plot_predictions(predictions[0], class_names, title="Ймовірності класів", color="skyblue")
                    st.pyplot(plt)  # Відображення графіка у Streamlit
                    predicted_label = class_names[np.argmax(predictions)]
                    st.write(f'Передбачення: {predicted_label}')

                except Exception as e:
                    st.error(f"Помилка обробки зображення для VGG16: {e}")

            else:
                st.warning("Оберіть валідну модель для прогнозування.")

                
    else:
        st.error("You are not logged in. Please login to see this page.")

