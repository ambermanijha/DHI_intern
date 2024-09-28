import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch.nn.functional as F

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("yanekyuk/bert-keyword-extractor")
model = AutoModelForTokenClassification.from_pretrained("yanekyuk/bert-keyword-extractor")

# Streamlit app layout
st.title("Keyword Extraction App")

# Create a form for user input
with st.form("keyword_form"):
    user_input = st.text_area("Enter your text:", "")
    submit = st.form_submit_button("Extract Keywords")

    if submit and user_input:
        # Preprocess the input text
        inputs = tokenizer.encode_plus(
            user_input,
            add_special_tokens=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Forward pass
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits

        # Get the predicted keywords
        predicted_keywords = torch.argmax(logits, dim=2)

        # Extract the keywords from the input text
        keywords = []
        for i, token in enumerate(predicted_keywords[0]):
            if token != 0:  # 0 is the padding token
                keywords.append(tokenizer.decode([token], skip_special_tokens=True))

        # Display extracted keywords
        if keywords:
            st.subheader("Extracted Keywords:")
            for keyword in keywords:
                st.write(keyword)
        else:
            st.write("No keywords found.")