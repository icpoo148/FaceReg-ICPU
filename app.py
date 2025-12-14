import streamlit as st
import face_recognition
import cv2
import numpy as np
from supabase import create_client, Client
import json
from datetime import datetime

# --- 1. SETUP CLOUD CONNECTION ---
# We get these secrets from Streamlit's secret manager (setup in next step)
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    st.error("Secrets not found! Please set up .streamlit/secrets.toml")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 2. DATABASE FUNCTIONS ---

def load_database():
    """Fetch all people from Supabase"""
    response = supabase.table('people').select("*").execute()
    data = response.data
    
    # Convert the stored string encoding back to numpy array
    for entry in data:
        entry['encoding'] = np.array(json.loads(entry['encoding']))
    return data

def save_to_cloud(name, role, company, image, encoding):
    """Upload image to Bucket and Data to Table"""
    
    # A. Upload Image
    filename = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    # Convert image to bytes for upload
    is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if is_success:
        file_bytes = buffer.tobytes()
        # Upload to 'photos' bucket
        supabase.storage.from_("photos").upload(filename, file_bytes, {"content-type": "image/jpeg"})
        
        # Get the Public URL so we can show it later
        image_url = supabase.storage.from_("photos").get_public_url(filename)
    else:
        st.error("Failed to process image")
        return

    # B. Upload Data
    data_entry = {
        "name": name,
        "role": role,
        "company": company,
        "image_url": image_url,
        "encoding": json.dumps(encoding.tolist()) # Convert Array -> JSON String
    }
    
    supabase.table('people').insert(data_entry).execute()

# --- 3. THE APP UI ---

st.set_page_config(page_title="WhoIsDat Cloud", layout="wide")
st.title("☁️ AI Person Recognizer (Online)")

tab1, tab2 = st.tabs(["➕ Add New Person", "🔍 Recognize from Photo"])

# TAB 1: ENROLLMENT
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        new_name = st.text_input("Name")
        new_role = st.text_input("Position")
        new_company = st.text_input("Company")
    with col2:
        ref_photo = st.file_uploader("Reference Photo", type=['jpg', 'png'])

    if st.button("Save to Cloud"):
        if new_name and ref_photo:
            with st.spinner("Uploading to cloud database..."):
                bytes_data = np.asarray(bytearray(ref_photo.read()), dtype=np.uint8)
                img = cv2.imdecode(bytes_data, 1)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                face_locs = face_recognition.face_locations(rgb_img)
                if len(face_locs) == 1:
                    encoding = face_recognition.face_encodings(rgb_img, face_locs)[0]
                    save_to_cloud(new_name, new_role, new_company, rgb_img, encoding)
                    st.success(f"✅ Saved {new_name} to Supabase!")
                else:
                    st.error("Please upload a photo with exactly one face.")

# TAB 2: RECOGNITION
with tab2:
    target_photo = st.file_uploader("Upload Group Photo", type=['jpg', 'png'])
    if target_photo:
        bytes_data = np.asarray(bytearray(target_photo.read()), dtype=np.uint8)
        img = cv2.imdecode(bytes_data, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(rgb_img, caption="Scanning...", use_column_width=True)
        
        if st.button("Identify"):
            with st.spinner("Fetching database and matching..."):
                db_data = load_database()
                known_encodings = [d['encoding'] for d in db_data]
                
                # Detect
                face_locs = face_recognition.face_locations(rgb_img)
                face_encodings = face_recognition.face_encodings(rgb_img, face_locs)
                
                st.write(f"Found {len(face_locs)} faces.")
                
                for (top, right, bottom, left), face_encoding in zip(face_locs, face_encodings):
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        person = db_data[best_match_index]
                        
                        # UI Card
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            # Show crop
                            face_crop = rgb_img[top:bottom, left:right]
                            st.image(face_crop, width=100)
                        with c2:
                            st.subheader(f"✅ {person['name']}")
                            st.write(f"**{person['company']}** - {person['role']}")
                            # Show the reference image from the URL
                            st.image(person['image_url'], width=80, caption="Ref from DB")
                        st.divider()