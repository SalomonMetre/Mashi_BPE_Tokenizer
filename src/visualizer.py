import streamlit as st
from Tokenizer import Tokenizer as MyTokenizer
import random
import colorsys # Standard library for color conversion

# Page config
st.set_page_config(page_title="Mashi Tokenizer Visualizer", layout="wide")

def get_token_color(token_id):
    """
    Generates high-contrast colors using HSL to ensure distinct boundaries
    while keeping background light enough for black text.
    """
    # Use the token ID to pick a hue (360 degrees)
    # Golden ratio shift helps spread colors as far apart as possible
    golden_ratio_conjugate = 0.618033988749895
    h = (token_id * golden_ratio_conjugate) % 1
    
    # S: 70-90% for vividness, L: 85-95% for light background
    s = 0.8
    l = 0.9 
    
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

@st.cache_resource
def load_tokenizer(vocab_path, merges_path):
    return MyTokenizer(vocab_path, merges_path)

def main():
    st.title("🧩 Custom BPE Tokenizer Visualizer")
    st.markdown("Visualizing subword segmentation for **Mashi** with high-contrast boundaries.")

    # Sidebar
    st.sidebar.header("Model Settings")
    vocab_file = st.sidebar.text_input("Vocab File", "tokenizer_files/vocab.txt")
    merges_file = st.sidebar.text_input("Merges File", "tokenizer_files/merges.txt")

    try:
        tokenizer = load_tokenizer(vocab_file, merges_file)
        st.sidebar.success("Tokenizer loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return

    user_input = st.text_area("Enter text to tokenize:", "moyo wa bantu banso bidi bimpe.", height=150)

    if user_input:
        ids = tokenizer.encode(user_input)
        
        col1, col2 = st.columns(2)
        col1.metric("Total Tokens", len(ids))
        col2.metric("Characters", len(user_input))
        
        st.subheader("Tokenized Result")
        
        # Build the HTML visualization with better styling
        html_output = (
            "<div style='line-height: 2.8; font-family: \"Source Code Pro\", monospace; "
            "font-size: 1.1rem; padding: 25px; border: 1px solid #ddd; border-radius: 12px; "
            "background-color: #ffffff; color: #000; box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>"
        )
        
        for token_id in ids:
            token_bytes = tokenizer.vocabulary[token_id]
            token_text = token_bytes.decode('utf-8', errors='replace')
            
            # Special handling for spaces to make them visible but distinct
            if token_text == " ":
                display_text = "&nbsp;"
                color = "#f0f0f0" # Neutral light gray for spaces
            else:
                display_text = token_text.replace("\n", "↵<br>")
                color = get_token_color(token_id)
            
            html_output += (
                f'<span style="background-color: {color}; padding: 5px 8px; margin: 2px; '
                f'border-radius: 6px; border: 1px solid rgba(0,0,0,0.15); font-weight: 500; '
                f'display: inline-block; transition: transform 0.1s;" '
                f'title="ID: {token_id}">{display_text}</span>'
            )
        
        html_output += "</div>"
        
        st.markdown(html_output, unsafe_allow_html=True)

        with st.expander("Show Raw IDs"):
            st.code(ids)

if __name__ == "__main__":
    main()