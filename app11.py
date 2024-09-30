import streamlit as st
import openai
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import time
import googlemaps
from streamlit_folium import folium_static
import folium
from streamlit_js_eval import get_geolocation
import google.generativeai as genai
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(page_title="Derm-AI Assistant", layout="wide")

# Custom CSS
st.markdown("""
<style>
    /* Change background color of the active tab */
    div[data-baseweb="tab"] {
        background-color: #f5f7fa;
    }
    
    /* Change background color of the selected tab */
    div[data-baseweb="tab"] button[aria-selected="true"] {
        background-color: #0073e6;
        color: white;
    }

    /* Change background color of unselected tabs */
    div[data-baseweb="tab"] button[aria-selected="false"] {
        background-color: #f0f2f6;
        color: #333;
    }

    /* Change hover effect on unselected tabs */
    div[data-baseweb="tab"] button[aria-selected="false"]:hover {
        background-color: #d6e4ff;
    }
</style>
""", unsafe_allow_html=True)


# Set up OpenAI API key

gmaps = googlemaps.Client(key="AIzaSyBZ54CrwbNjBiKKs-4NydriYQTp0yEGFlM")
# Load the model
model = load_model('weights.h5')

genai.configure(api_key="AIzaSyA8CHnU_1P-UMjwR9bK9Fn77zmymPNXC5Y")

def preprocess_image(image):
    img = image.resize((128, 128))  # Resize the image
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)

def get_chatbot_response(user_input):
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])
    prompt = f"""
    You are a medical chatbot specializing in cancer treatment, diagnosis, and prevention. 
    Provide concise answers, but elaborate when necessary for complex topics. 
    Always prioritize accurate medical information and encourage users to consult healthcare professionals for personalized advice.

    """
    response = chat.send_message(user_input)
    response_text = response.text.replace("**", "").replace("\n\n", "\n").strip()
    return response_text

def get_nearby_hospitals(lat, lng):
    # Search for nearby cancer hospitals
    places_result = gmaps.places_nearby(
        location=(lat, lng),
        radius=10000,  # 10km in meters
        keyword='cancer hospital'
    )
    
    hospitals = places_result.get('results', [])[:10]  # Limit to top 10 results
    
    # Get additional details for each hospital
    for hospital in hospitals:
        place_id = hospital['place_id']
        details = gmaps.place(place_id, fields=['formatted_phone_number', 'website'])
        hospital['phone_number'] = details['result'].get('formatted_phone_number', 'N/A')
        hospital['website'] = details['result'].get('website', 'N/A')
    
    return hospitals

# Function to get a readable location name
def get_location_name(lat, lng):
    result = gmaps.reverse_geocode((lat, lng))
    if result:
        # Try to get the most specific address component
        for component in result[0]['address_components']:
            if 'sublocality' in component['types']:
                return component['long_name']
        # If no sublocality, try to get the locality
        for component in result[0]['address_components']:
            if 'locality' in component['types']:
                return component['long_name']
        # If no specific component found, return the formatted address
        return result[0]['formatted_address']
    return "Unknown location"

# Streamlit app layout
def create_pdf_report(patient_info, image, prediction, class_label):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add patient information
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Patient Report")
    c.setFont("Helvetica", 12)
    y = height - 80
    for key, value in patient_info.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20

    # Add image classification results
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y - 20, "Image Classification Results")
    c.setFont("Helvetica", 12)
    c.drawString(50, y - 40, f"Prediction: {class_label}")
    c.drawString(50, y - 60, f"Confidence Score: {prediction[0][0]:.2f}")

    # Add the image
    img_width, img_height = image.size
    aspect = img_height / float(img_width)
    max_width = 400
    max_height = 300
    display_width = min(max_width, img_width)
    display_height = min(max_height, int(display_width * aspect))
    c.drawImage(ImageReader(image), 50, y - 70 - display_height, width=display_width, height=display_height)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ... (keep the existing functions) ...

# Streamlit app layout
st.title("Derm-AI Assistant")
st.write("Cancer Information and Diagnosis Assistant")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Image Classification", "Chatbot", "Nearby Hospitals"])

# ... (previous code remains unchanged)

with tab1:
    st.header("Skin Cancer Image Classification")
    
    # Patient information form
    st.subheader("Patient Information")
    patient_name = st.text_input("Name")
    patient_age = st.number_input("Age", min_value=0, max_value=120)
    patient_sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    patient_contact = st.text_input("Contact Details")

    uploaded_files = st.file_uploader("Choose images for skin cancer classification:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        if len(uploaded_files) < 1 or len(uploaded_files) > 3:
            st.warning("Please upload between 1 to 3 images.")
        else:
            for index, uploaded_file in enumerate(uploaded_files):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
                
                with col2:
                    # Preprocess the image
                    image = Image.open(uploaded_file)
                    processed_image = preprocess_image(image)

                    # Make predictions
                    prediction = model.predict(processed_image)
                    class_label = "Potentially Malignant" if prediction[0] > 0.5 else "Likely Not Malignant"

                    # Display the result
                    st.write(f"Prediction: {class_label}")
                    st.write(f"Confidence Score: {(prediction[0][0]*100):.2f}%")
                    st.write("Note: This is a preliminary assessment. Please consult a dermatologist for a professional diagnosis.")

                    # Create and offer PDF download
                    if st.button("Generate PDF Report", key=f"generate_pdf_{index}"):
                        patient_info = {
                            "Name": patient_name,
                            "Age": patient_age,
                            "Sex": patient_sex,
                            "Contact": patient_contact
                        }
                        pdf_buffer = create_pdf_report(patient_info, image, prediction, class_label)
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"patient_report_{index}.pdf",
                            mime="application/pdf",
                            key=f"download_pdf_{index}"
                        )
    else:
        st.info("Please upload at least one image for skin cancer classification.")

# ... (rest of the code remains unchanged)

with tab2:
    st.header("Medical Chatbot")
    user_input = st.text_input("Ask a question about cancer treatment, diagnosis, or prevention:", "")

    if user_input:
        response = get_chatbot_response(user_input)
        st.text_area("Medical Chatbot:", value=response, height=200, max_chars=None, key=None)




with tab3:
    st.header("Find Nearby Cancer Hospitals")
    loc = get_geolocation()

    if loc:
        lat = loc['coords']['latitude']
        lng = loc['coords']['longitude']
        location_name = get_location_name(lat, lng)
        
        st.write(f"Your location: {location_name}")
        
        hospitals = get_nearby_hospitals(lat, lng)
        if hospitals:
            st.write(f"Found {len(hospitals)} cancer hospitals within 10km radius.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Nearest 10 Hospitals")
                for i, hospital in enumerate(hospitals, 1):
                    with st.expander(f"{i}. {hospital['name']}"):
                        st.write(f"Phone: {hospital['phone_number']}")
                        if hospital['website'] != 'N/A':
                            st.write(f"Website: [{hospital['website']}]({hospital['website']})")
            
            with col2:
                m = folium.Map(location=[lat, lng], zoom_start=12)
                
                folium.Marker(
                    location=[lat, lng],
                    popup=f"Your Location: {location_name}",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
                
                for hospital in hospitals:
                    folium.Marker(
                        location=[hospital['geometry']['location']['lat'], hospital['geometry']['location']['lng']],
                        popup=hospital['name'],
                        tooltip=hospital['name']
                    ).add_to(m)
                
                folium_static(m)
        else:
            st.warning("No cancer hospitals found within 10km radius.")
    else:
        st.warning("Unable to get your location. Please make sure you've granted location access to this site.")

# Sidebar with Skin Cancer Information
st.sidebar.title("Skin Cancer Awareness")
st.sidebar.info("""
**Common Symptoms of Skin Cancer:**
- Unusual moles or changes in existing moles
- Sores that don't heal
- Pigmented patches or growths on the skin
- Lumps or spots that itch, crust, or bleed

**Preventive Measures:**
1. Use broad-spectrum sunscreen (SPF 30+)
2. Avoid tanning beds
3. Wear protective clothing and seek shade
4. Perform regular skin self-exams
5. Consult a dermatologist for suspicious skin changes

Early detection is crucial. If you notice any concerning changes, consult a healthcare professional immediately.
""")

# Footer
st.markdown("---")
st.markdown("© 2024 Derm-AI. This app is for informational purposes only and does not substitute professional medical advice.")
