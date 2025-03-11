Image Steganography With CNN based Encoder-Decoder Model 

This project is a web-based application for hiding secret messages within digital images using a neural network-based encoder-decoder model. The application allows users to upload an image, embed a secret message, and generate a stego image that is visually indistinguishable from the original. The hidden message can later be extracted by authorized users. The project is built using Django for the backend, TensorFlow for the neural network, and SQLite for database management.

Features

User Authentication:

Users can register and log in to the system.
Secure password storage using Django's built-in hashing.
Encoding:

Users can upload an image and embed a secret message (up to 20 characters).
The system uses a neural network to encode the message into the image.
Decoding:

Users can upload a stego image and extract the hidden message.
The system uses a neural network to decode the message from the image.
Email Integration:

Users can send the stego image to a recipient via email.
The system uses Gmail's SMTP server for sending emails.
User-Friendly Interface:

The application provides an intuitive and responsive interface using HTML, CSS, and JavaScript.
Algorithms and Technologies

Neural Network:

A simple encoder-decoder architecture is used for encoding and decoding.
The encoder embeds the secret message into the image.
The decoder extracts the hidden message from the stego image.
The model is trained using TensorFlow and Keras.
Image Processing:

OpenCV is used for image preprocessing (resizing, normalization).
Matplotlib is used for image visualization.
Web Framework:

Django is used for the backend and web interface.
Database:

SQLite is used for storing user and decoder information.
Email Integration:

Django's email backend is used for sending emails via Gmail's SMTP server.
Prerequisites

Before running the project, ensure you have the following installed:

Python 3.7 or higher.
Django.
TensorFlow.
OpenCV.
Matplotlib.
Gmail Account (for email integration).
Steps for Project Execution

Clone the Repository:

git clone https://github.com/your-username/image-steganography.git
cd image-steganography
** Install Dependencies:**

pip install -r requirements.txt
Set Up the Database:

python manage.py migrate
Run the Development Server:

python manage.py runserver
Access the Application: Open your browser and go to http://127.0.0.1:8000/.

Send Emails: Update the EMAIL_HOST_USER and EMAIL_HOST_PASSWORD in settings.py with your Gmail credentials.

Project Workflow

User Registration: Users register by providing their full name, email, and password.

User Login: Registered users log in using their email and password.

Encoding: Users upload an image and enter a secret message. The system encodes the message into the image and generates a stego image.

Decoding: Users upload a stego image. The system decodes the hidden message and displays it to the user.

Email Integration: Users send the stego image to a recipient via email.
