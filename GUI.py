# gui.py
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
from transformer_engine import IndianHistoryTransformer, prepare_tensors, MAX_LEN
from history_data import get_data
from keras.utils import pad_sequences

st.set_page_config(page_title="Indian History Transformer", layout="wide")
st.title("🕉️ Indian History Transformer (Scratch Implementation)")

# --- 1. LOAD DATA ---
texts, labels, label_map = get_data()
st.sidebar.subheader("Training Data Sample")
st.sidebar.write(texts[:3])

# --- 2. PREPARE TENSORS ---
X, tokenizer = prepare_tensors(texts)
y = torch.tensor(labels, dtype=torch.long)

# --- 3. INIT MODEL ---
if 'model' not in st.session_state:
    st.session_state.model = IndianHistoryTransformer()
    st.session_state.trained = False

model = st.session_state.model

# --- GUI SECTION: TRAINING ---
st.subheader("1. Train the Transformer")
st.write("Train the manual encoder stack on the history dataset and observe the loss slope.")

epochs = st.slider("Epochs", 10, 100, 50)
lr = st.select_slider("Learning Rate", options=[0.01, 0.001, 0.0001], value=0.001)

if st.button("Start Training"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    progress_bar = st.progress(0)
    chart_place = st.empty()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        # Update Chart Live
        if epoch % 5 == 0:
            df = pd.DataFrame({'Training Loss': loss_history})
            chart_place.line_chart(df)
            progress_bar.progress((epoch + 1) / epochs)
            time.sleep(0.05) # Just for visual effect
            
    st.success(f"Training Complete! Final Loss: {loss_history[-1]:.4f}")
    st.session_state.trained = True

# --- GUI SECTION: INFERENCE ---
st.divider()
st.subheader("2. Query the Model (RAG Context)")

user_query = st.text_input("Enter a history snippet (e.g., 'system of drainage'):")

if st.button("Predict Era"):
    if not st.session_state.trained:
        st.warning("Please train the model first!")
    else:
        # Preprocess single input
        model.eval()
        seq = tokenizer.texts_to_sequences([user_query])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        input_tensor = torch.tensor(padded, dtype=torch.long)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        
        st.markdown(f"### Predicted Context: **{label_map[prediction]}**")
        st.write(f"Confidence: {confidence:.2%}")
        st.code(f"Logits: {logits.numpy()}", language="python")