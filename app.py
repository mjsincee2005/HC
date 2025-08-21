import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ================================
# Load the trained model
# ================================
MODEL_PATH = r"C:\CODING\AI\PROJECTS\HC\Models\VGGnet\vgg_hair_disease_model.h5"  # Change to your .h5 file path
model = load_model(MODEL_PATH)

# ================================
# Class labels in the same order as your training
# ================================
CLASS_NAMES = [
    'Alopecia Areata',
    'Contact Dermatitis',
    'Folliculitis',
    'Head Lice',
    'Healthy hair',
    'Lichen Planus',
    'Male Pattern Baldness',
    'Psoriasis',
    'Seborrheic Dermatitis',
    'Telogen Effluvium',
    'Tinea Capitis'
]

# ================================
# Disease details dictionary
# ================================
DISEASE_INFO = {
    'Alopecia Areata': {
        'description': 'An autoimmune condition causing patchy hair loss.',
        'symptoms': 'Round or oval patches of hair loss, smooth scalp in affected areas',
        'treatment': 'Corticosteroid injections, topical treatments, immunotherapy'
    },
    'Contact Dermatitis': {
        'description': 'Skin inflammation caused by contact with an irritant or allergen.',
        'symptoms': 'Redness, itching, swelling, blisters on the scalp',
        'treatment': 'Avoid triggers, topical corticosteroids, antihistamines'
    },
    'Folliculitis': {
        'description': 'Inflammation of hair follicles, often caused by bacterial infection.',
        'symptoms': 'Small red bumps, pus-filled bumps around hair follicles',
        'treatment': 'Topical antibiotics, warm compresses, good hygiene'
    },
    'Head Lice': {
        'description': 'Parasitic insects that live on the human scalp.',
        'symptoms': 'Intense itching, visible lice or nits, red bumps on scalp',
        'treatment': 'Medicated shampoos, fine-toothed comb, thorough cleaning'
    },
    'Healthy hair': {
        'description': 'Normal, healthy hair and scalp condition.',
        'symptoms': 'No visible signs of disease or damage',
        'treatment': 'Maintain good hair care routine and healthy lifestyle'
    },
    'Lichen Planus': {
        'description': 'Inflammatory condition affecting skin and hair follicles.',
        'symptoms': 'Purple, itchy, flat-topped bumps, possible hair loss',
        'treatment': 'Topical corticosteroids, oral medications, light therapy'
    },
    'Male Pattern Baldness': {
        'description': 'Genetic hair loss pattern common in men.',
        'symptoms': 'Receding hairline, crown thinning, gradual hair loss',
        'treatment': 'Minoxidil, finasteride, hair transplant procedures'
    },
    'Psoriasis': {
        'description': 'Chronic autoimmune condition causing skin cell buildup.',
        'symptoms': 'Red, scaly patches, silvery scales, itching',
        'treatment': 'Topical treatments, light therapy, systemic medications'
    },
    'Seborrheic Dermatitis': {
        'description': 'Common skin condition causing scaly, itchy patches.',
        'symptoms': 'Flaky, white or yellow scales, redness, itching',
        'treatment': 'Medicated shampoos, topical antifungals, corticosteroids'
    },
    'Telogen Effluvium': {
        'description': 'Temporary hair loss due to stress or medical conditions.',
        'symptoms': 'Diffuse hair thinning, increased hair shedding',
        'treatment': 'Address underlying cause, nutritional support, patience'
    },
    'Tinea Capitis': {
        'description': 'Fungal infection of the scalp and hair.',
        'symptoms': 'Scaly patches, broken hairs, possible bald spots',
        'treatment': 'Oral antifungal medications, medicated shampoos'
    }
}

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Hair Disease Detection", layout="centered")
st.title("💇 Hair Disease Detection App")
st.write("Upload an image of the scalp/hair to detect possible disease and get details.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image (adjust size to match model input)
    img_resized = img.resize((224, 224))  # Change size if your model uses different input
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show results
    st.subheader(f"🩺 Predicted Disease: {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Show disease details
    info = DISEASE_INFO[predicted_class]
    st.markdown(f"**Description:** {info['description']}")
    st.markdown(f"**Symptoms:** {info['symptoms']}")
    st.markdown(f"**Treatment:** {info['treatment']}")
