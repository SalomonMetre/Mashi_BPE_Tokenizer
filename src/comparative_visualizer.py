import sys
import os
sys.path.append(os.path.join(os.getcwd()))

import streamlit as st
from Tokenizer import Tokenizer as MyTokenizer
import random
import tiktoken
from transformers import AutoTokenizer
import colorsys

# Page config
st.set_page_config(page_title="Mashi Tokenizer: Fertility Analysis", layout="wide")

def get_token_color(token_id, seed_offset=0):
    golden_ratio_conjugate = 0.618033988749895
    h = ((token_id + seed_offset) * golden_ratio_conjugate) % 1
    s = 0.8  
    l = 0.9  
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

@st.cache_resource
def load_custom_tokenizer(vocab_path, merges_path):
    return MyTokenizer(vocab_path, merges_path)

@st.cache_resource
def load_industry_tokenizers():
    gpt = tiktoken.get_encoding("o200k_base")
    llama = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    return gpt, llama

def render_tokenized_html(tokens_list, model_name, seed_offset, fertility, token_ids=None):
    html_output = (
        f"<div style='line-height: 2.5; font-family: monospace; font-size: 1rem; "
        f"padding: 20px; border: 1px solid #ddd; border-radius: 12px; "
        f"background-color: #ffffff; margin-bottom: 25px; color: #000; "
        f"box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>"
    )
    # Highlighted Fertility in the header of each block
    html_output += (
        f"<div style='margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 5px;'>"
        f"<span style='color: #333; font-weight: bold;'>{model_name}</span>"
        f"<span style='color: #007bff; font-weight: bold; font-size: 0.9rem;'>Fertility: {fertility:.2f}</span>"
        f"</div>"
    )
    
    for i, token_text in enumerate(tokens_list):
        color_key = token_ids[i] if token_ids else i
        color = get_token_color(color_key, seed_offset)
        display_text = token_text.replace(" ", "&nbsp;").replace("\n", "↵<br>")
        
        html_output += (
            f'<span style="background-color: {color}; padding: 4px 6px; margin: 2px; '
            f'border-radius: 5px; border: 1px solid rgba(0,0,0,0.1); '
            f'display: inline-block; font-weight: 500;">{display_text}</span>'
        )
    
    html_output += "</div>"
    return html_output

def main():
    # UPDATED TITLE
    st.title("📊 Tokenizer Fertility Analysis")
    st.markdown("Comparing **Mashi BPE** tokenizer against LLMs' tokenizers")

    st.sidebar.header("Custom Model Settings")
    vocab_file = st.sidebar.text_input("Vocab File", "tokenizer_files/vocab.txt")
    merges_file = st.sidebar.text_input("Merges File", "tokenizer_files/merges.txt")

    try:
        my_tokenizer = load_custom_tokenizer(vocab_file, merges_file)
        gpt_tok, llama_tok = load_industry_tokenizers()
        st.sidebar.success("Models loaded.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        return

    user_input = st.text_area("Input Text (in Mashi):", "moyo wa bantu banso bidi bimpe.", height=100)

    if user_input:
        word_count = len(user_input.split()) or 1
        
        # 1. Custom
        my_ids = my_tokenizer.encode(user_input)
        my_tokens = [my_tokenizer.vocabulary[i].decode('utf-8', errors='replace') for i in my_ids]
        my_fertility = len(my_ids) / word_count

        # 2. GPT-4o
        gpt_ids = gpt_tok.encode(user_input)
        gpt_tokens = [gpt_tok.decode([i]) for i in gpt_ids]
        gpt_fertility = len(gpt_ids) / word_count

        # 3. Llama 3
        llama_ids = llama_tok.encode(user_input, add_special_tokens=False)
        llama_tokens = [llama_tok.decode([i]) for i in llama_ids]
        llama_fertility = len(llama_ids) / word_count

        # BIG FERTILITY METRICS
        cols = st.columns(3)
        cols[0].metric("Custom BPE Fertility", f"{my_fertility:.2f}", delta=f"{len(my_ids)} tokens", delta_color="inverse")
        cols[1].metric("GPT-4o Fertility", f"{gpt_fertility:.2f}", delta=f"{len(gpt_ids)} tokens", delta_color="off")
        cols[2].metric("Llama 3 Fertility", f"{llama_fertility:.2f}", delta=f"{len(llama_ids)} tokens", delta_color="off")

        st.divider()
        
        # 
        
        # RENDER BLOCKS
        st.markdown(render_tokenized_html(my_tokens, "Custom Mashi BPE", 100, my_fertility, token_ids=my_ids), unsafe_allow_html=True)
        st.markdown(render_tokenized_html(gpt_tokens, "GPT-4o (tiktoken)", 200, gpt_fertility), unsafe_allow_html=True)
        st.markdown(render_tokenized_html(llama_tokens, "Llama 3 (Meta)", 300, llama_fertility), unsafe_allow_html=True)

        with st.expander("📚 What is Fertility?"):
            st.write("""
            **Fertility** is the average number of tokens per word. 
            - **Low Fertility (~1.0):** The tokenizer identifies whole words or large morphemes. This is highly efficient.
            - **High Fertility (>2.5):** The tokenizer is splitting words into many small pieces (sometimes single bytes), which usually happens when the model hasn't seen the language often enough.
            """)

if __name__ == "__main__":
    main()