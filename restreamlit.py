import os
os.environ["SENTENCE_TRANSFORMERS_NO_TF"] = "1"
# research_navigator_streamlit.py
import streamlit as st
import spacy
from PyPDF2 import PdfReader
import re
import html
import numpy as np
from collections import Counter, defaultdict
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.models import Model
import tensorflow as tf
import math
import string
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



st.set_page_config(layout="wide", page_title="Research Navigator")
nlp = spacy.load("en_core_web_sm")

ENTITY_COLORS = {
    "PERSON": "#ffadad", "ORG": "#ffd6a5", "GPE": "#caffbf",
    "DATE": "#9bf6ff", "WORK_OF_ART": "#bdb2ff", "MONEY": "#ffc6ff",
    "PRODUCT": "#ffffba", "EVENT": "#ffd6e0", "LOC": "#caffbf",
    "NORP": "#d0f4de", "FAC": "#ffd7a8", "LAW": "#c0f0c0",
    "LANGUAGE": "#f9f0ff", "TIME": "#e0fbfc", "PERCENT": "#ffe8a1",
    "QUANTITY": "#c6dbf5", "CARDINAL": "#e7c6ff", "ORDINAL": "#f6d6d6",
    "DEFAULT": "#E6E6FA"
}

# Utility Functions

def read_pdf(file):
    try:
        reader = PdfReader(file)
        texts = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
        return " ".join(texts)
    except:
        return ""

def clean_text(text: str) -> str:
    t = text.replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def segment_sentences(text: str):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def tokenize_text(text: str):
    doc = nlp(text)
    return [token.text for token in doc]

def extract_entities_spacy(text: str):
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} for ent in doc.ents]

def noun_phrases(text: str, top_k=30):
    doc = nlp(text)
    phrases = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip())>1]
    freq = Counter(phrases)
    return [p for p,_ in freq.most_common(top_k)]

METHOD_VERBS = ["propose","proposes","proposed","design","develop","use","utilize","train","evaluate","implement","apply","introduce","introduces","present"]
FINDING_KEYWORDS = ["achieve","achieves","achieved","outperform","outperforms","improve","improves","improved","show","shows","demonstrate","demonstrates","result","results","higher","lower","increase","decrease","accuracy","mAP","F1","precision","recall"]

def extract_methodologies(sentences):
    methods = []
    for s in sentences:
        s_l = s.lower()
        if any(re.search(r'\b' + mv + r'\b', s_l) for mv in METHOD_VERBS):
            methods.append(s.strip())
    return methods

def extract_findings(sentences):
    findings = []
    for s in sentences:
        s_l = s.lower()
        if any(re.search(r'\b' + fk + r'\b', s_l) for fk in FINDING_KEYWORDS):
            findings.append(s.strip())
    return findings

def render_colored_tokens(tokens, token_labels, semantic_colors):
    out = "<div style='line-height:2.2;padding:6px'>"
    for i, tok in enumerate(tokens):
        label = token_labels[i] if i < len(token_labels) and token_labels[i] else ""
        color = semantic_colors.get(i, "#FFFFFF")
        tooltip = html.escape(label if label else "Token")
        out += f"<span title='{tooltip}' style='background:{color};padding:6px;margin:3px;border-radius:6px;display:inline-block'>{html.escape(tok)}</span>"
    out += "</div>"
    return out

def build_concept_graph(node_embeddings, node_labels, threshold=0.6):
    G = nx.Graph()
    n = len(node_labels)
    for i in range(n):
        G.add_node(i, label=node_labels[i])
    if n <= 1: return G
    sims = cosine_similarity(node_embeddings)
    for i in range(n):
        for j in range(i+1, n):
            score = sims[i,j]
            if score >= threshold:
                G.add_edge(i,j, weight=float(score))
    return G

def graph_to_plotly(G, labels, node_colors=None, node_sizes=None):
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for u,v in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        edge_x.extend([x0,x1,None]); edge_y.extend([y0,y1,None])
    edge_trace = go.Scatter(x=edge_x,y=edge_y,mode='lines',line=dict(width=1,color='#888'),hoverinfo='none')
    node_x,node_y,node_text = [],[],[]
    node_color_list=[]; node_size_list=[]
    for n in G.nodes():
        x,y = pos[n]
        node_x.append(x); node_y.append(y)
        node_text.append(labels[n])
        node_color_list.append(node_colors[n] if node_colors else "#66b3ff")
        node_size_list.append(node_sizes[n] if node_sizes else 14)
    node_trace = go.Scatter(x=node_x,y=node_y,mode='markers+text',text=node_text,textposition="top center",
                            marker=dict(size=node_size_list,color=node_color_list,line=dict(width=1,color='#333')))
    fig = go.Figure(data=[edge_trace,node_trace])
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10),showlegend=False)
    return fig

# ------------------------------
# App UI
# ------------------------------
st.title("Research Navigator — NER, Key Concepts, Methods, Findings, Graph & Chatbot")

# Sidebar input controls
with st.sidebar:
    st.header("Input / Options")
    input_mode = st.radio("Input type", ["Paste paragraph", "Upload files (pdf/txt)"])
    docs = []
    doc_titles = []

    if input_mode == "Paste paragraph":
        user_text = st.text_area("Paste your paragraph(s) here:", height=250)
        if user_text and user_text.strip():
            docs = [user_text.strip()]
            doc_titles = ["User Paragraph"]
    else:
        uploaded = st.file_uploader("Upload multiple PDF / TXT", accept_multiple_files=True, type=['pdf','txt'])
        if uploaded:
            for f in uploaded:
                content = ""
                if f.name.lower().endswith(".pdf") or f.type == "application/pdf":
                    content = read_pdf(f)
                else:
                    try:
                        content = f.read().decode("utf-8")
                    except:
                        content = str(f.read())
                docs.append(content)
                doc_titles.append(f.name)

    st.markdown("---")
    st.subheader("Graph & Embeddings")
    edge_threshold = st.slider("Graph similarity threshold", 0.4, 0.95, 0.65, 0.01)
    n_token_clusters = st.number_input("Token semantic clusters (for coloring)", min_value=2, max_value=20, value=8)

    st.markdown("---")
    st.subheader("Generator (Model Controls)")

    
    model_choice = st.selectbox(
        "Select generator model (CPU-friendly)",
        options=[
            "google/flan-t5-small",
            "t5-small",
            "google/flan-t5-base"
        ],
        index=0
    )

    max_len = st.number_input(
        "Max generation tokens",
        min_value=32,
        max_value=512,
        value=150
    )

    num_beams = st.slider("num_beams (if not sampling)", 1, 8, 4)
    do_sample = st.checkbox("Use sampling (stochastic outputs, slower)", value=False)
    temperature = st.slider("Temperature (when sampling)", 0.1, 1.5, 0.7)
    top_k = st.number_input("top_k (when sampling)", min_value=10, max_value=200, value=50)
    top_p = st.slider("top_p (when sampling)", 0.1, 1.0, 0.95)

    st.markdown("---")

# Stop execution if no documents provided
if not docs:
    st.info("Please paste text or upload files in the sidebar to proceed.")
    st.stop()

# Combine and clean text
merged = " \n ".join([d for d in docs if d and d.strip()])
cleaned = clean_text(merged)
# Split cleaned text into sentences

doc_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned) if s.strip()]


st.header("Preprocessing preview")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Raw (first 700 chars)")
    st.write(merged[:700] + ("..." if len(merged) > 700 else ""))
    st.subheader("Cleaned (first 700 chars)")
    st.write(cleaned[:700] + ("..." if len(cleaned) > 700 else ""))
with col2:
    st.subheader("Segmentation (first 8 sentences)")
    sents = segment_sentences(cleaned)
    for i, s in enumerate(sents[:8]):
        st.markdown(f"{i+1}. {s}")
    st.subheader("Tokenization (first 40 tokens)")
    tokens = tokenize_text(cleaned)
    st.write(tokens[:40])

# NER extraction & token-level labels
st.header("Named Entity Recognition (NER) — colored token boxes")
ents = extract_entities_spacy(cleaned)
token_labels = [""] * len(tokens)
cursor = 0; token_positions = []
for tok in tokens:
    idx = cleaned.find(tok, cursor)
    if idx == -1: token_positions.append((-1,-1)); continue
    token_positions.append((idx, idx+len(tok))); cursor = idx+len(tok)
for i,(s,e) in enumerate(token_positions):
    if s==-1: continue
    for ent in ents:
        if s<ent["end"] and e>ent["start"]: token_labels[i] = ent["label"]; break

# Embeddings
st.header("Embeddings & Semantic Clustering (token & sentence)")
sentences_for_tokenizer = sents if len(sents)>0 else [cleaned]
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences_for_tokenizer)
vocab_size = min(20000, len(tokenizer.word_index)+1)
maxlen = min(100, max(1, max(len(tokenizer.texts_to_sequences([s])[0]) for s in sentences_for_tokenizer)))
st.write(f"Vocab used: {vocab_size}, maxlen (for seqs): {maxlen}")

embedding_dim = 128
idx_in = Input(shape=(1,), dtype='int32')
emb_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name="emb_layer")
emb_out = emb_layer(idx_in)
emb_model = Model(idx_in, emb_out)

token_indices_seq = tokenizer.texts_to_sequences(tokens)
token_idx_flat = [seq[0] if len(seq)>0 else 0 for seq in token_indices_seq]
token_idx_arr = np.array(token_idx_flat)
token_embs_raw = emb_model.predict(token_idx_arr, verbose=0)
token_embs = token_embs_raw.reshape((token_embs_raw.shape[0], token_embs_raw.shape[2]))
token_embs_norm = normalize(token_embs)

sent_embeddings = []
for sent in sents:
    seq = tokenizer.texts_to_sequences([sent])[0]
    if len(seq)==0: sent_embeddings.append(np.zeros((embedding_dim,)))
    else:
        inds = [i if i < vocab_size else 0 for i in seq]
        emb = emb_model.predict(np.array(inds), verbose=0).reshape((len(inds), embedding_dim))
        sent_embeddings.append(np.mean(emb, axis=0))
sent_embeddings = normalize(np.array(sent_embeddings)) if len(sent_embeddings)>0 else np.array([])

st.success("Created lightweight token & sentence embeddings (replace with trained Seq2Seq encoder for stronger semantics).")

# Semantic clustering
n_clusters = min(n_token_clusters, max(2, int(math.sqrt(len(token_embs_norm)))))
clusterer = AgglomerativeClustering(n_clusters=n_clusters)
cluster_labels = clusterer.fit_predict(token_embs_norm)
palette = ["#FFB3BA","#FFDFBA","#FFFFBA","#BAFFC9","#BAE1FF","#D9B3FF","#FFB3E6","#B3FFD9","#B3D9FF","#E6B3FF"]
cluster_colors = {i: palette[i % len(palette)] for i in range(n_clusters)}
semantic_colors = {i: cluster_colors[cluster_labels[i]] for i in range(len(tokens))}
for i,lab in enumerate(token_labels):
    if lab: semantic_colors[i] = ENTITY_COLORS.get(lab, ENTITY_COLORS["DEFAULT"])
st.markdown(render_colored_tokens(tokens[:400], token_labels[:400], semantic_colors), unsafe_allow_html=True)

# Key Concepts, Methods, Findings
st.header("Extracted Insights")
entities_texts = [e["text"].strip() for e in ents]
top_entities = list(dict(Counter([t.lower() for t in entities_texts])).keys())[:30]
top_noun_phrases = noun_phrases(cleaned, top_k=40)
key_concepts = []
seen = set()
for item in top_entities + top_noun_phrases:
    key = item.strip()
    if key.lower() not in seen:
        key_concepts.append(item)
        seen.add(key.lower())
if len(key_concepts)==0: key_concepts = top_noun_phrases[:20]
methods = extract_methodologies(sents)
findings = extract_findings(sents)
colA, colB, colC = st.columns(3)
with colA:
    st.subheader("Key Concepts")
    for kc in key_concepts[:30]: st.markdown(f"- {kc}")
with colB:
    st.subheader("Research Methodologies (extracted)")
    if methods:
        for m in methods[:20]: st.markdown(f"- {m}")
    else: st.info("No obvious methodology sentences found by rule-based patterns.")
with colC:
    st.subheader("Findings / Results (extracted)")
    if findings:
        for f in findings[:20]: st.markdown(f"- {f}")
    else: st.info("No clear findings detected by rule-based patterns.")

# Graph
st.header("Semantic Graph: Concepts, Methods, Findings")
nodes = key_concepts + methods + findings
node_types = ["concept"]*len(key_concepts) + ["method"]*len(methods) + ["finding"]*len(findings)
seen = set(); uniq_nodes = []; uniq_types=[]
for n,t in zip(nodes,node_types):
    key = n.strip().lower()
    if key not in seen: uniq_nodes.append(n); uniq_types.append(t); seen.add(key)
nodes = uniq_nodes; node_types = uniq_types
if nodes:
    node_embs = []
    for n in nodes:
        seq = tokenizer.texts_to_sequences([n])[0]
        if len(seq)==0: node_embs.append(np.zeros((embedding_dim,)))
        else:
            inds = [i if i < vocab_size else 0 for i in seq]
            emb = emb_model.predict(np.array(inds), verbose=0).reshape((len(inds), embedding_dim))
            node_embs.append(np.mean(emb, axis=0))
    node_embs = normalize(np.array(node_embs)) if len(node_embs)>0 else node_embs
    G = build_concept_graph(node_embs, nodes, threshold=edge_threshold)
    type_color_map = {"concept":"#66b3ff","method":"#9bf6ff","finding":"#ffd6a5"}
    node_colors = [type_color_map.get(t,"#cccccc") for t in node_types]
    freq = Counter([tok.lower() for tok in tokens])
    node_sizes = [8 + math.log(1+sum(freq.get(w,0) for w in re.findall(r"\w+", n.lower()))+0.0001)*8 for n in nodes]
    fig = graph_to_plotly(G, nodes, node_colors=node_colors, node_sizes=node_sizes)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Node Inspector")
    colI1, colI2 = st.columns([2,3])
    with colI1:
        choice_mode = st.radio("Select node by", ["Index","Label"])
        if choice_mode=="Index":
            idx = st.number_input("Node index", min_value=0, max_value=max(0,len(nodes)-1), value=0, step=1)
            selected_node = nodes[idx]; selected_type = node_types[idx]
        else:
            sel_label = st.selectbox("Choose node label", nodes)
            sel_idx = nodes.index(sel_label); idx = sel_idx
            selected_node = sel_label; selected_type = node_types[sel_idx]
    with colI2:
        st.markdown(f"**Selected Node:** {selected_node}")
        st.markdown(f"**Type:** {selected_type}")
        occurrences = [(i,s) for i,s in enumerate(sents) if any(w.lower() in s.lower() for w in re.findall(r"\w+", selected_node))]
        st.markdown(f"**Occurrences (first 8):**")
        for occ in occurrences[:8]: st.markdown(f"- Sent {occ[0]}: {occ[1][:250]}...")
        sims = cosine_similarity([node_embs[idx]], node_embs)[0]
        topk = np.argsort(sims)[-8:][::-1]
        st.markdown("**Top semantic neighbors:**")
        for t in topk:
            if t==idx: continue
            st.markdown(f"- {nodes[t]} (sim={sims[t]:.3f})")

# Chatbot Section — Sentence Embeddings + RAG
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.markdown("---")
st.header("Ask Questions / Summarize Document (RAG - Sentence Embeddings + Flan-T5)")

# Sentence Embedding Model 
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
doc_embeddings = embedder.encode(doc_sentences, normalize_embeddings=True)

#  Generator Model (Flan-T5 or T5) 
@st.cache_resource(show_spinner=False)
def load_generator(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model, device

tokenizer, model, device = load_generator(model_choice)

#Retrieval Function (semantic search)
def retrieve_semantic_chunks(query, doc_embeddings, docs, top_k=3):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q_emb, doc_embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(docs[i], float(sims[i])) for i in top_idx if sims[i] > 0.2]

# Prompt Build
def build_prompt(query, retrieved):
    context = "\n\n".join([chunk for chunk, _ in retrieved])
    return f"Use the below research context to answer the query.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

#  Answer Generator 
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_len,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams if not do_sample else 1,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Streamlit Chat UI 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_message = st.text_input("Ask your question or type 'summary' / 'short notes':")

if user_message:
    st.session_state.chat_history.append(("user", user_message))
    query = user_message.strip().lower()

    # Handle summarization requests
    if query in ("summary", "short notes", "notes", "gist", "outline"):
        context = " ".join(doc_sentences[:15])
        prompt = f"Summarize the following research content:\n\n{context}"
        answer = generate_answer(prompt)
        retrieved = []
    else:
        retrieved = retrieve_semantic_chunks(query, doc_embeddings, doc_sentences, top_k=5)
        if not retrieved:
            answer = "Sorry, I couldn’t find relevant context to answer that."
        else:
            prompt = build_prompt(query, retrieved)
            answer = generate_answer(prompt)

    st.session_state.chat_history.append(("assistant", answer))
    st.session_state.last_retrieved = retrieved

# Display Chat History
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)

#  Show Retrieved Context
st.markdown("### Retrieved Context")
if "last_retrieved" in st.session_state and st.session_state.last_retrieved:
    for i, (chunk, score) in enumerate(st.session_state.last_retrieved, start=1):
        with st.expander(f"Chunk {i} (similarity: {score:.3f})"):
            st.write(chunk)
else:
    st.info("No retrieved chunks yet.")


