import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from supabase import create_client, Client
import json
from datetime import datetime
import uuid
import time

# --- 1. SETUP CLOUD CONNECTION ---
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
    return response.data

def save_to_cloud(name, role, company, image, embedding):
    """Upload image to Bucket and Data to Table"""
    
    # Generate a random safe filename
    safe_filename = f"{uuid.uuid4()}.jpg"
    
    # A. Upload Image
    is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    if is_success:
        try:
            file_bytes = buffer.tobytes()
            supabase.storage.from_("photos").upload(safe_filename, file_bytes, {"content-type": "image/jpeg"})
            image_url = supabase.storage.from_("photos").get_public_url(safe_filename)
        except Exception as e:
            st.error(f"Image Upload Failed: {e}")
            return False
    else:
        st.error("Failed to process image")
        return False

    # B. Upload Data
    data_entry = {
        "name": name,
        "role": role,
        "company": company,
        "image_url": image_url,
        "encoding": json.dumps(embedding)
    }
    
    try:
        response = supabase.table('people').insert(data_entry).execute()
        return True
    except Exception as e:
        st.error(f"Database Save Failed: {e}. (Check RLS settings!)")
        return False

# --- 3. THE APP UI ---

st.set_page_config(page_title="WhoIsDat Cloud", layout="wide")
st.title("☁️ AI Person Recognizer (DeepFace)")

tab1, tab2 = st.tabs(["➕ Add New Person", "🔍 Recognize from Photo"])

# ==========================================
# TAB 1: ENROLLMENT (Manual Add Only)
# ==========================================
with tab1:
    st.header("Register New Person")
    col1, col2 = st.columns(2)
    with col1:
        new_name = st.text_input("Name")
        new_role = st.text_input("Position")
        new_company = st.text_input("Company")
    with col2:
        ref_photo = st.file_uploader("Reference Photo", type=['jpg', 'png', 'jpeg'])

    if st.button("Save to Cloud"):
        if new_name and ref_photo:
            with st.spinner("Processing AI & Uploading..."):
                bytes_data = np.asarray(bytearray(ref_photo.read()), dtype=np.uint8)
                img = cv2.imdecode(bytes_data, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                try:
                    embedding = DeepFace.represent(img_path=img_rgb, model_name="VGG-Face", enforce_detection=True)[0]["embedding"]
                    
                    success = save_to_cloud(new_name, new_role, new_company, img_rgb, embedding)
                    if success:
                        st.success(f"✅ Saved {new_name} to database!")
                        time.sleep(1)
                        st.rerun()
                        
                except ValueError:
                    st.error("❌ No face detected. Please use a clear photo.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================================
# TAB 2: RECOGNITION ONLY (No Add)
# ==========================================
with tab2:
    st.header("Scan Group Photo")
    target_photo = st.file_uploader("Upload Group Photo", type=['jpg', 'png', 'jpeg'])
    
    if target_photo:
        bytes_data = np.asarray(bytearray(target_photo.read()), dtype=np.uint8)
        img = cv2.imdecode(bytes_data, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        st.image(img_rgb, caption="Scanning...", use_column_width=True)
        
        if st.button("Identify People"):
            with st.spinner("Analyzing faces..."):
                db_data = load_database()
                
                try:
                    faces = DeepFace.extract_faces(img_path=img_rgb, detector_backend='opencv', enforce_detection=False)
                    st.write(f"Found **{len(faces)}** faces.")
                    st.divider()
                    
                    for i, face_obj in enumerate(faces):
                        detected_face = face_obj["face"]
                        display_face = (detected_face * 255).astype(np.uint8)
                        
                        try:
                            current_embedding = DeepFace.represent(img_path=display_face, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
                        except:
                            continue 

                        best_match = None
                        lowest_distance = 1.0 
                        
                        for person in db_data:
                            db_emb = json.loads(person['encoding'])
                            a = np.array(current_embedding)
                            b = np.array(db_emb)
                            dist = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                            
                            if dist < lowest_distance:
                                lowest_distance = dist
                                best_match = person
                        
                        # --- DISPLAY RESULT CARD ---
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            st.image(display_face, width=120, caption=f"Face #{i+1}")
                        
                        with c2:
                            # MATCH FOUND (Threshold 0.40)
                            if lowest_distance < 0.40:
                                st.subheader(f"✅ {best_match['name']}")
                                st.write(f"🏢 **{best_match['company']}**")
                                st.write(f"📋 {best_match['role']}")
                                st.caption(f"Confidence: {((1-lowest_distance)*100):.1f}%")
                                if best_match.get('image_url'):
                                    st.image(best_match['image_url'], width=60)
                                    
                            # UNKNOWN (Just show text, NO ADD FORM)
                            else:
                                st.subheader("❓ Unknown Person")
                                st.warning("Not found in database.")

                        st.divider()
                        
                except Exception as e:
                    st.error(f"Could not process image. Error: {e}")
