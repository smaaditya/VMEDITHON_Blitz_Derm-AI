import streamlit as st
import google.generativeai as genai
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import googlemaps
from streamlit_folium import folium_static
import folium
from streamlit_js_eval import get_geolocation
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.units import inch
from datetime import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

# Set page config at the very beginning
st.set_page_config(page_title="Derm-AI Assistant", layout="wide")

# Custom CSS
st.markdown("""
<style>
    div[data-baseweb="tab"] {
        background-color: #f5f7fa;
    }
    div[data-baseweb="tab"] button[aria-selected="true"] {
        background-color: #0073e6;
        color: white;
    }
    div[data-baseweb="tab"] button[aria-selected="false"] {
        background-color: #f0f2f6;
        color: #333;
    }
    div[data-baseweb="tab"] button[aria-selected="false"]:hover {
        background-color: #d6e4ff;
    }
</style>
""", unsafe_allow_html=True)

# Set up API keys and clients
gmaps = googlemaps.Client(key="AIzaSyDD3k1fw2nnrIpZY-Lq17fJS6rB6ibvNmM")
genai.configure(api_key="AIzaSyA8CHnU_1P-UMjwR9bK9Fn77zmymPNXC5Y")

# Load the model
model = load_model('weights.h5')

# Set up the OAuth 2.0 flow
client_config = {
    "web": {
        "client_id": st.secrets["GOOGLE_CLIENT_ID"],
        "project_id": st.secrets["GOOGLE_PROJECT_ID"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
        "redirect_uris": [st.secrets["REDIRECT_URI"]]
    }
}

flow = Flow.from_client_config(
    client_config,
    scopes=['https://www.googleapis.com/auth/userinfo.email', 'https://www.googleapis.com/auth/userinfo.profile'],
    redirect_uri=st.secrets["REDIRECT_URI"]
)

# Helper functions
def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_chatbot_response(user_input):
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])
    prompt = f"""
    You are a medical chatbot specializing in cancer treatment, diagnosis, and prevention. 
    Provide concise answers, but elaborate when necessary for complex topics. 
    Always prioritize accurate medical information and encourage users to consult healthcare professionals for personalized advice.
    User query: {user_input}
    """
    response = chat.send_message(prompt)
    return response.text.replace("**", "").replace("\n\n", "\n").strip()

def get_nearby_hospitals(lat, lng):
    places_result = gmaps.places_nearby(
        location=(lat, lng),
        radius=10000,
        keyword='cancer hospital'
    )
    
    hospitals = places_result.get('results', [])[:10]
    
    for hospital in hospitals:
        place_id = hospital['place_id']
        details = gmaps.place(place_id, fields=['formatted_phone_number', 'website'])
        hospital['phone_number'] = details['result'].get('formatted_phone_number', 'N/A')
        hospital['website'] = details['result'].get('website', 'N/A')
    
    return hospitals

def get_location_name(lat, lng):
    result = gmaps.reverse_geocode((lat, lng))
    if result:
        for component in result[0]['address_components']:
            if 'sublocality' in component['types']:
                return component['long_name']
        for component in result[0]['address_components']:
            if 'locality' in component['types']:
                return component['long_name']
        return result[0]['formatted_address']
    return "Unknown location"

def create_pdf_report(patient_info, image, prediction, class_label):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add header
    c.setFillColor(colors.navy)
    c.rect(0, height - 50, width, 50, fill=True)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 35, "Derm-AI Assistant Report")

    # Add report date
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(width - 200, height - 65, f"Report Date: {report_date}")

    # Add patient information
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 100, "Patient Information")
    
    data = [[key, value] for key, value in patient_info.items()]
    table = Table(data, colWidths=[100, 200])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
    ]))
    table.wrapOn(c, width, height)
    table.drawOn(c, 50, height - 250)

    # Add image classification results
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 280, "Image Classification Results")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 300, f"Prediction: {class_label}")
    c.drawString(50, height - 320, f"Confidence Score: {(prediction[0][0]*100):.2f}%")

    # Add the image
    img_width, img_height = image.size
    aspect = img_height / float(img_width)
    max_width = 400
    max_height = 300
    display_width = min(max_width, img_width)
    display_height = min(max_height, int(display_width * aspect))
    c.drawImage(ImageReader(image), 50, height - 320 - display_height, width=display_width, height=display_height)

    # Add footer
    c.setFillColor(colors.grey)
    c.rect(0, 0, width, 30, fill=True)
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 8)
    c.drawString(50, 10, "© 2024 Derm-AI. This report is for informational purposes only and does not substitute professional medical advice.")
    c.drawString(width - 200, 10, f"Page 1 of 1")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def display_user_info(credentials):
    try:
        service = build('oauth2', 'v2', credentials=credentials)
        user_info = service.userinfo().get().execute()
        st.sidebar.write(f"Welcome, {user_info['name']}!")
        st.sidebar.write(f"Email: {user_info['email']}")
        st.sidebar.image(user_info['picture'], width=100)
    except Exception as e:
        st.sidebar.error(f"An error occurred while fetching user info: {e}")

def main():
    st.title("Derm-AI Assistant")
    st.write("Cancer Information and Diagnosis Assistant")

    # Check if the user is already authenticated
    if 'credentials' in st.session_state:
        credentials = Credentials(**st.session_state['credentials'])
        if credentials and credentials.valid:
            display_user_info(credentials)
            if st.sidebar.button("Logout"):
                del st.session_state['credentials']
                st.experimental_rerun()
        else:
            del st.session_state['credentials']
            st.experimental_rerun()
    else:
        # If not authenticated, show the login button
        if 'login_button_clicked' not in st.session_state:
            st.session_state.login_button_clicked = False
    
        if st.sidebar.button("Login with Google", key="login_button") or st.session_state.login_button_clicked:
            st.session_state.login_button_clicked = True
            authorization_url, _ = flow.authorization_url(prompt='consent')
            st.sidebar.markdown(f'<a href="{authorization_url}" target="_self">Click here to login</a>', unsafe_allow_html=True)

    # Check for the authorization response
    params = st.experimental_get_query_params()
    if 'code' in params:
        try:
            flow.fetch_token(code=params['code'][0])
            credentials = flow.credentials
            st.session_state['credentials'] = {
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes
            }
            st.experimental_set_query_params()
            st.experimental_rerun()
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Main application content
    if 'credentials' in st.session_state:
        tab1, tab2, tab3 = st.tabs(["Image Classification", "Chatbot", "Nearby Hospitals"])

        with tab1:
            st.header("Skin Cancer Image Classification")
            
            # Patient information form
            with st.form("patient_info"):
                st.subheader("Patient Information")
                patient_name = st.text_input("Name")
                patient_age = st.number_input("Age", min_value=0, max_value=120)
                patient_sex = st.selectbox("Sex", ["Male", "Female", "Other"])
                patient_contact = st.text_input("Contact Details")
                patient_id = st.text_input("Patient ID (optional)")
                submit_button = st.form_submit_button("Submit")

            if submit_button:
                st.success("Patient information submitted successfully!")

            uploaded_files = st.file_uploader("Choose images for skin cancer classification:", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        if len(uploaded_files) < 1 or len(uploaded_files) > 3:
            st.warning("Please upload between 1 to 3 images.")
        else:
            for i, uploaded_file in enumerate(uploaded_files):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
                
                with col2:
                    try:
                        image = Image.open(uploaded_file)
                        processed_image = preprocess_image(image)
                        prediction = model.predict(processed_image)
                        class_label = "Potentially Malignant" if prediction[0] > 0.5 else "Likely Not Malignant"
    
                        st.write(f"Prediction: {class_label}")
                        st.write(f"Confidence Score: {(prediction[0][0]*100):.2f}%")
                        st.write("Note: This is a preliminary assessment. Please consult a dermatologist for a professional diagnosis.")
    
                        if st.button(f"Generate PDF Report for {uploaded_file.name}", key=f"generate_pdf_{i}"):
                            patient_info = {
                                "Name": patient_name,
                                "Age": str(patient_age),
                                "Sex": patient_sex,
                                "Contact": patient_contact,
                                "Patient ID": patient_id if patient_id else "N/A"
                            }
                            pdf_buffer = create_pdf_report(patient_info, image, prediction, class_label)
                            st.download_button(
                                label=f"Download PDF Report for {uploaded_file.name}",
                                data=pdf_buffer,
                                file_name=f"patient_report_{patient_name.replace(' ', '_')}_{i}.pdf",
                                mime="application/pdf",
                                key=f"download_pdf_{i}"
                            )
                    except Exception as e:
                        st.error(f"An error occurred during processing: {str(e)}")
    else:
        st.info("Please upload at least one image for skin cancer classification.")

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
                            with st.expander(f"{i}. {hospital['name']}", key=f"hospital_{i}"):

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

if __name__ == "__main__":
    main()
