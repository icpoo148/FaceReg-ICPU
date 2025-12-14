import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from supabase import create_client, Client
import json
from datetime import datetime
import uuid

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
    
    # Generate a random safe filename (prevents errors with special characters)
    safe_filename = f"{uuid.uuid4()}.jpg"
    
    # A. Upload Image
    # Convert numpy image to bytes
    is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    if is_success:
        file_bytes = buffer.tobytes()
        # Upload to Supabase Storage
        supabase.storage.from_("photos").upload(safe_filename, file_bytes, {"content-type": "image/jpeg"})
        # Get the Public URL
        image_url = supabase.storage.from_("photos").get_public_url(safe_filename)
    else:
        st.error("Failed to process image")
        return

    # B. Upload Data
    data_entry = {
        "name": name,
        "role": role,
        "company": company,
        "image_url": image_url,
        "encoding": json.dumps(embedding) # Store AI vector as JSON string
    }
    
    supabase.table('people').insert(data_entry).execute()

# --- 3. THE APP UI ---

st.set_page_config(page_title="WhoIsDat Cloud", layout="wide")
st.title("☁️ AI Person Recognizer (DeepFace)")

tab1, tab2 = st.tabs(["➕ Add New Person", "🔍 Recognize from Photo"])

# ==========================================
# TAB 1: ENROLLMENT (Manual Add)
# ==========================================
with tab1:
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
                # Prepare Image
                bytes_data = np.asarray(bytearray(ref_photo.read()), dtype=np.uint8)
                img = cv2.imdecode(bytes_data, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                try:
                    # Generate AI Embedding
                    embedding = DeepFace.represent(img_path=img_rgb, model_name="VGG-Face", enforce_detection=True)[0]["embedding"]
                    
                    save_to_cloud(new_name, new_role, new_company, img_rgb, embedding)
                    st.success(f"✅ Saved {new_name} to database!")
                except ValueError:
                    st.error("❌ No face detected. Please use a clear photo.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================================
# TAB 2: RECOGNITION & QUICK ADD
# ==========================================
with tab2:
    target_photo = st.file_uploader("Upload Group Photo", type=['jpg', 'png', 'jpeg'])
    
    if target_photo:
        bytes_data = np.asarray(bytearray(target_photo.read()), dtype=np.uint8)
        img = cv2.imdecode(bytes_data, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        st.image(img_rgb, caption="Scanning...", use_column_width=True)
        
        if st.button("Identify People"):
            with st.spinner("Analyzing faces..."):
                # 1. Fetch Database
                db_data = load_database()
                
                try:
                    # 2. Extract Faces from Group Photo
                    # 'opencv' backend is faster and more stable on cloud than dlib
                    faces = DeepFace.extract_faces(img_path=img_rgb, detector_backend='opencv', enforce_detection=False)
                    st.write(f"Found **{len(faces)}** faces.")
                    
                    # 3. Process Each Face found
                    for i, face_obj in enumerate(faces):
                        detected_face = face_obj["face"]
                        # Convert float image (0-1) to int (0-255) for display
                        display_face = (detected_face * 255).astype(np.uint8)
                        
                        # Get embedding for this specific crop
                        try:
                            current_embedding = DeepFace.represent(img_path=display_face, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
                        except:
                            continue # Skip if AI cannot read the face

                        # 4. Compare with Database
                        best_match = None
                        lowest_distance = 1.0 # 1.0 = different, 0.0 = same
                        
                        for person in db_data:
                            db_emb = json.loads(person['encoding'])
                            # Cosine Distance
                            a = np.array(current_embedding)
                            b = np.array(db_emb)
                            dist = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                            
                            if dist < lowest_distance:
                                lowest_distance = dist
                                best_match = person
                        
                        # 5. Display Result Card
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            st.image(display_face, width=120, caption=f"Face #{i+1}")
                        
                        with c2:
                            # --- MATCH FOUND (Threshold 0.40) ---
                            if lowest_distance < 0.40:
                                st.subheader(f"✅ {best_match['name']}")
                                st.write(f"🏢 **{best_match['company']}**")
                                st.write(f"📋 {best_match['role']}")
                                st.caption(f"Confidence: {((1-lowest_distance)*100):.1f}%")
                                if best_match.get('image_url'):
                                    st.image(best_match['image_url'], width=60, caption="Reference")
                                    
                            # --- UNKNOWN (Quick Add Form) ---
                            else:
                                st.subheader("❓ Unknown Person")
                                st.warning("Not in database")
                                
                                # Quick Add Form
                                with st.expander(f"➕ Add Person #{i+1} to DB"):
                                    with st.form(key=f"add_form_{i}"):
                                        st.caption("Add details for this face:")
                                        u_name = st.text_input("Name")
                                        u_role = st.text_input("Role")
                                        u_comp = st.text_input("Company")
                                        
                                        if st.form_submit_button("Save to Database"):
                                            if u_name:
                                                save_to_cloud(u_name, u_role, u_comp, display_face, current_embedding)
                                                st.success(f"Saved {u_name}! Re-scan to see them.")
                                            else:
                                                st.error("Name is required.")

                        st.divider()
                        
                except Exception as e:
                    st.error(f"Could not process image. Error: {e}")
