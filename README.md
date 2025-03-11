# Image Steganography With CNN based Encoder-Decoder Model

This project is a **web-based application** for hiding secret messages within digital images using a **neural network-based encoder-decoder model**. The application allows users to upload an image, embed a secret message, and generate a stego image that is visually indistinguishable from the original. The hidden message can later be extracted by authorized users. The project is built using **Django** for the backend, **TensorFlow** for the neural network, and **SQLite** for database management.

---

## **Features**
1. **User Authentication**:
   - Users can register and log in to the system.
   - Secure password storage using Django's built-in hashing.

2. **Encoding**:
   - Users can upload an image and embed a secret message (up to 20 characters).
   - The system uses a neural network to encode the message into the image.

3. **Decoding**:
   - Users can upload a stego image and extract the hidden message.
   - The system uses a neural network to decode the message from the image.

4. **Email Integration**:
   - Users can send the stego image to a recipient via email.
   - The system uses Gmail's SMTP server for sending emails.

5. **User-Friendly Interface**:
   - The application provides an intuitive and responsive interface using **HTML**, **CSS**, and **JavaScript**.

---

## **Algorithms and Technologies**
1. **Neural Network**:
   - A **simple encoder-decoder architecture** is used for encoding and decoding.
   - The encoder embeds the secret message into the image.
   - The decoder extracts the hidden message from the stego image.
   - The model is trained using **TensorFlow** and **Keras**.

2. **Image Processing**:
   - **OpenCV** is used for image preprocessing (resizing, normalization).
   - **Matplotlib** is used for image visualization.

3. **Web Framework**:
   - **Django** is used for the backend and web interface.

4. **Database**:
   - **SQLite** is used for storing user and decoder information.

5. **Email Integration**:
   - **Django's email backend** is used for sending emails via Gmail's SMTP server.

---

## **Prerequisites**
Before running the project, ensure you have the following installed:
1. **Python 3.7 or higher**.
2. **Django**.
3. **TensorFlow**.
4. **OpenCV**.
5. **Matplotlib**.
6. **Gmail Account** (for email integration).

---

## **Steps for Project Execution**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/image-steganography.git
   cd image-steganography
   
2. ** Install Dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Set Up the Database:**
   ```bash
   python manage.py migrate

4. **Run the Development Server:**
   ```bash
   python manage.py runserver

5. **Access the Application:**
Open your browser and go to http://127.0.0.1:8000/.

6. **Send Emails:**
Update the EMAIL_HOST_USER and EMAIL_HOST_PASSWORD in settings.py with your Gmail credentials.

---

## **Project Workflow**
1. **User Registration:**
Users register by providing their full name, email, and password.

2. **User Login:**
Registered users log in using their email and password.

3. **Encoding:**
Users upload an image and enter a secret message.
The system encodes the message into the image and generates a stego image.

4. **Decoding:**
Users upload a stego image.
The system decodes the hidden message and displays it to the user.

5. **Email Integration:**
Users send the stego image to a recipient via email.
