# =============================================================================
# helper.py - Shared functions for SDG Analysis System
# =============================================================================

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, BertTokenizerFast, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import PyPDF2
import spacy
import coreferee
import nltk
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import ipywidgets as widgets
from nltk.tokenize import sent_tokenize
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Dict, Any
from spacy.lang.en.stop_words import STOP_WORDS


# =============================================================================
# Model Initialization (Shared)
# =============================================================================

# Load spaCy + coreferee
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('coreferee')
# Send everything to CUDA when available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT classification model
BERT_CHECKPOINT = "sadickam/sdg-classification-bert"
config = AutoConfig.from_pretrained(BERT_CHECKPOINT)
config.architectures = ["BertForSequenceClassification"]

bert_tokenizer = BertTokenizerFast.from_pretrained(
    BERT_CHECKPOINT, use_fast=True)
bert_model = BertForSequenceClassification.from_pretrained(
    BERT_CHECKPOINT,
    config=config,
    trust_remote_code=True
)
bert_model.to(device).eval()

# Load MPNet similarity model
SIM_CHECKPOINT = "sentence-transformers/paraphrase-mpnet-base-v2"
sim_tokenizer = AutoTokenizer.from_pretrained(SIM_CHECKPOINT)
sim_model = SentenceTransformer(SIM_CHECKPOINT)
sim_model.to(device).eval()

# SDG target descriptions
SDG_TARGETS = [
    "No Poverty", "Zero Hunger", "Good Health and Well-being", "Quality Education",
    "Gender Equality", "Clean Water and Sanitation", "Affordable and Clean Energy",
    "Decent Work and Economic Growth", "Industry, Innovation and Infrastructure",
    "Reduced Inequality", "Sustainable Cities and Communities",
    "Responsible Consumption and Production", "Climate Action", "Life Below Water",
    "Life on Land", "Peace, Justice and Strong Institutions"
]

# =============================================================================
# PDF Extraction Functions
# =============================================================================


def extract_text_from_pdf_robust(pdf_path):
    """Robust PDF text extraction with error handling"""
    print(f"📄 Extracting text from PDF: {pdf_path}")

    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            if pdf_reader.is_encrypted:
                print("🔒 PDF is encrypted, attempting to decrypt...")
                pdf_reader.decrypt('')

            text = ""
            pages_extracted = 0

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += page_text + "\n\n"
                        pages_extracted += 1
                        print(
                            f"  ✅ Page {page_num + 1}: {len(page_text)} characters")
                except Exception as e:
                    print(f"  ⚠️  Page {page_num + 1}: {e}")
                    continue

            print(
                f"✅ Extracted {pages_extracted}/{len(pdf_reader.pages)} pages, {len(text)} characters")
            return text.strip()

    except Exception as e:
        raise Exception(f"PDF extraction failed: {e}")


def preprocess_pdf_text(text, max_length=None, return_paragraphs=False):
    """Clean and preprocess extracted PDF text"""
    print("🧹 Preprocessing PDF text…")

    # Merge hyphenated line-breaks
    text = re.sub(r'-\s*\n\s*', '', text)

    # Temporarily mark true paragraph breaks
    text = re.sub(r'\n\s*\n+', '<PARA>', text)

    # Collapse any other newlines into spaces
    text = re.sub(r'\n+', ' ', text)

    # Restore paragraph breaks
    text = text.replace('<PARA>', '\n\n')

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text).strip()

    # Split into paragraphs and filter
    paras = text.split('\n\n')
    cleaned_paras = []
    for para in paras:
        p = para.strip()
        if len(p) < 20 or p.isdigit():
            continue
        cleaned_paras.append(p)

    # Reconstruct full text
    cleaned_text = '\n\n'.join(cleaned_paras)

    # Truncate if needed
    if max_length and len(cleaned_text) > max_length:
        snippet = cleaned_text[:max_length]
        cut = snippet.rfind('.')
        cleaned_text = (snippet[:cut+1] if cut >
                        max_length * 0.8 else snippet).strip()
        print(f"⚠️  Text truncated to {len(cleaned_text)} characters")
        cleaned_paras = cleaned_text.split('\n\n')

    print(
        f"✅ Text preprocessing complete: {len(cleaned_text)} characters, {len(cleaned_paras)} paragraphs")
    return cleaned_paras if return_paragraphs else cleaned_text

# =============================================================================
# Sentence Filtering Functions
# =============================================================================


def is_citation_or_reference(sentence):
    """Enhanced citation and reference detection"""
    citation_patterns = [
        r'\([^)]*\d{4}[^)]*\)', r'\[\d+\]', r'\b[A-Z][a-z]+\s+et\s+al\.',
        r'^\s*\d+\.', r'^\s*[A-Z]\.', r'doi\.org|DOI:|http://|https://|www\.',
        r'^\s*Fig\.|^\s*Figure|^\s*Table|^\s*Eq\.',
        r'See\s+(Fig|Figure|Table|Section|Appendix)',
        r'^\s*References?\s*$', r'^\s*Bibliography\s*$'
    ]
    return any(re.search(pattern, sentence, re.IGNORECASE) for pattern in citation_patterns)


def has_meaningful_content(sentence):
    """Check if sentence has meaningful content"""
    cleaned = re.sub(r'\([^)]*\d{4}[^)]*\)', '', sentence)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    words = cleaned.split()
    meaningful_words = [w for w in words if len(w) > 2 and not w.isdigit()]

    if len(meaningful_words) < 5:
        return False

    common_verbs = [
        'is', 'are', 'was', 'were', 'has', 'have', 'had', 'can', 'could', 'will', 'would',
        'show', 'shows', 'indicate', 'suggest', 'demonstrate', 'reveal', 'provide', 'present'
    ]

    has_verb = any(verb in cleaned.lower() for verb in common_verbs)
    ends_properly = sentence.strip().endswith(('.', '!', '?', ';'))

    return has_verb and ends_properly


def extract_and_filter_sentences(text):
    """Extract and filter sentences from text"""
    print("✂️  Extracting and filtering sentences...")

    raw_sentences = sent_tokenize(text)
    filtered_sentences = []

    for sentence in raw_sentences:
        sentence = re.sub(r'\s+', ' ', sentence).strip()

        if (len(sentence) < 20 or
            is_citation_or_reference(sentence) or
            not has_meaningful_content(sentence) or
                len(sentence.split()) < 6):
            continue

        filtered_sentences.append(sentence)

    print(
        f"✅ Filtered {len(filtered_sentences)} valid sentences from {len(raw_sentences)} raw")
    return filtered_sentences


def prep_text(text):
    """Text preprocessing for BERT"""
    clean_sents = []
    sent_tokens = sent_tokenize(str(text))
    for sent_token in sent_tokens:
        word_tokens = [str(word_token).strip().lower()
                       for word_token in sent_token.split()]
        clean_sents.append(' '.join(word_tokens))
    joined = ' '.join(clean_sents).strip(' ')
    joined = re.sub(r'`', "", joined)
    joined = re.sub(r'"', "", joined)
    return joined

# =============================================================================
# Classification Functions
# =============================================================================


def classify_sentence_bert(sentence):
    """Classify sentence using BERT model"""
    inputs = bert_tokenizer(sentence, return_tensors="pt",
                            truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = bert_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze(0)
        label = int(torch.argmax(probs))
        conf = float(probs[label])
    return label, conf


def classify_sentence_similarity(sentence, similarity_threshold=0.6):
    """Compute semantic similarity label and score using SentenceTransformer"""
    sent_emb = sim_model.encode(sentence, convert_to_tensor=True, device=device)
    target_embs = sim_model.encode(SDG_TARGETS, convert_to_tensor=True, device=device   )

    sims = util.cos_sim(sent_emb, target_embs)[0]
    best_score = float(sims.max())
    best_idx = int(sims.argmax())

    if best_score < similarity_threshold:
        return None, best_score
    return best_idx, best_score


def determine_dual_consensus(bert_label, bert_conf, sim_label, sim_conf,
                             agree_thresh=0.1, disagree_thresh=0.2, min_conf=0.5):
    """Determine consensus between BERT and similarity approaches"""
    if sim_label is None or sim_label < 0:
        return 'bert_only' if bert_conf >= min_conf else 'mistrust'
    if bert_conf < min_conf and sim_conf < min_conf:
        return 'mistrust'
    if bert_label == sim_label and bert_conf >= min_conf and sim_conf >= min_conf:
        return 'agreement'
    diff = abs(bert_conf - sim_conf)
    if diff >= disagree_thresh:
        return 'bert_only' if bert_conf > sim_conf else 'similarity_only'
    if bert_label != sim_label and diff >= agree_thresh:
        return 'disagreement'
    if bert_conf >= min_conf:
        return 'bert_only'
    if sim_conf >= min_conf:
        return 'similarity_only'
    return 'disagreement'

# =============================================================================
# Coreference Analysis Functions
# =============================================================================


def expand_to_full_phrase(text, char_start, char_end):
    """Expand span to full phrase using dependency parsing"""
    doc = nlp(text)
    target_token = None

    for token in doc:
        if token.idx <= char_start < token.idx + len(token.text):
            target_token = token
            break

    if target_token is None:
        return text[char_start:char_end], (char_start, char_end)

    # Get the subtree
    subtree = list(target_token.subtree)
    start = subtree[0].idx
    end = subtree[-1].idx + len(subtree[-1].text)

    # Try to extend for modifiers
    sent = target_token.sent
    i = subtree[-1].i + 1
    while i < len(doc) and doc[i].idx < sent.end_char:
        if doc[i].dep_ in ("prep", "amod", "advmod", "compound", "pobj"):
            end = doc[i].idx + len(doc[i].text)
        else:
            break
        i += 1

    return text[start:end], (start, end)


def analyze_full_text_coreferences(full_text):
    """Analyze coreferences using coreferee API"""
    doc = nlp(full_text)

    if not hasattr(doc._, 'coref_chains'):
        return {"chains": [], "error": "Coreferee not properly loaded"}

    chains_data = []
    for chain_idx, chain in enumerate(doc._.coref_chains):
        chain_mentions = []

        for mention in chain:
            try:
                if hasattr(mention, 'token_indexes'):
                    token_indices = mention.token_indexes
                else:
                    token_indices = list(mention)

                mention_tokens = [doc[i]
                                  for i in token_indices if 0 <= i < len(doc)]
                if not mention_tokens:
                    continue

                mention_text = " ".join([t.text for t in mention_tokens])
                start_char = mention_tokens[0].idx
                end_char = mention_tokens[-1].idx + \
                    len(mention_tokens[-1].text)

                chain_mentions.append({
                    "text": mention_text,
                    "start_char": start_char,
                    "end_char": end_char,
                    "token_indices": token_indices
                })

            except Exception as e:
                continue

        if chain_mentions:
            representative = max(
                chain_mentions, key=lambda x: len(x["text"]))["text"]
            chains_data.append({
                "chain_id": chain_idx,
                "representative": representative,
                "mentions": chain_mentions
            })

    return {"chains": chains_data}


def normalize_span_for_chaining(sentence, local_start, local_end):
    """Generate candidate spans for coreference matching"""
    doc = nlp(sentence)
    span_text = sentence[local_start:local_end]
    span_doc = nlp(span_text)
    candidates = []

    # COMPOUND‐NOUN AWARENESS
    for tok in span_doc:
        if tok.pos_ == "NOUN" and tok.dep_ == "compound":
            for nc in doc.noun_chunks:
                if nc.start <= tok.i < nc.end:
                    # full compound phrase
                    candidates.append((nc.text, nc.start_char, nc.end_char))
                    # head noun of that compound
                    head = tok.head
                    candidates.append(
                        (head.text, head.idx, head.idx + len(head.text)))
                    break

    # Rule 1: If span lacks any NOUN, expand to nearest noun phrase
    if not any(t.pos_ == "NOUN" for t in span_doc):
        anchor = next((t for t in doc if t.idx == local_start), None)
        if anchor:
            left = anchor
            while left.i > 0 and left.pos_ != "NOUN":
                left = doc[left.i - 1]
            for nc in doc.noun_chunks:
                if nc.start <= left.i < nc.end:
                    candidates.append((nc.text, nc.start_char, nc.end_char))
                    break

    # Rule 2: If span has no subject/object or root not noun, include subj & obj heads
    root = next((tok for tok in span_doc if tok.dep_ == 'ROOT'), None)
    if root:
        sent = root.sent
        subj = next((t for t in sent if t.dep_ == 'nsubj'), None)
        obj = next((t for t in sent if t.dep_ in ('dobj', 'obj')), None)
        if root.pos_ != 'NOUN' or not ({t.dep_ for t in span_doc} & {'nsubj', 'dobj', 'obj'}):
            if subj:
                candidates.append(
                    (subj.text, subj.idx, subj.idx + len(subj.text)))
            if obj:
                candidates.append(
                    (obj.text,  obj.idx,  obj.idx + len(obj.text)))

    # Rule 3: If span is mostly stopwords, strip them and retry
    words = span_text.split()
    if words and all(w.lower() in STOP_WORDS for w in words[:2] + words[-2:]):
        stripped = ' '.join([w for w in words if w.lower() not in STOP_WORDS])
        pos = sentence.find(stripped)
        if stripped:
            # use spaCy to get exact char offset inside sentence
            stripped_doc = nlp(stripped)
            if stripped_doc:
                s_off = stripped_doc[0].idx
                e_off = s_off + len(stripped)
                candidates.append((stripped, s_off, e_off))
    # Rule 4 & 5: For any noun_chunks inside the span, add full chunk + each noun token
    for nc in span_doc.noun_chunks:
        # full chunk
        s_abs = local_start + nc.start_char
        e_abs = local_start + nc.end_char
        candidates.append((nc.text, s_abs, e_abs))
        # each noun in the chunk
        for tok in nc:
            if tok.pos_ == 'NOUN':
                # token.idx is offset in 'sentence'
                s_off = tok.idx
                e_off = s_off + len(tok.text)
                candidates.append((tok.text, s_off, e_off))
    # Always include the original span
    candidates.append((span_text, local_start, local_end))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for text, s, e in candidates:
        key = (text, s, e)
        if key not in seen:
            seen.add(key)
            unique.append((text, s, e))
    return unique

# =============================================================================
# Visualization Functions
# =============================================================================
# File: helper.py

import string
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# reuse a single SentenceTransformer instance
_model = SentenceTransformer('all-MiniLM-L6-v2')
PRONOUNS = {'it','its','they','them','their','itself'}

def list_and_filter_coref_clusters_from_kpe(sentence_analyses):
    """
    Build clusters from each keyphrase/span coreference_analysis.
    Keeps only chains that link across ≥2 sentences & have ≥1 non-pronoun mention.
    Expects sentence_analyses = [
      {
        'sentence_id': int,
        'sentence_text': str,
        'keyphrase_analysis' or 'span_analysis': List[{
            'phrase' or 'span': str,
            'expanded_phrase': str,
            'coreference_analysis': {
               'chain_found': bool,
               'chain_id': int,
               …
            }
        }]
      }, …
    ]
    Returns list of {chain_id, mentions, sentences}.
    """
    raw = {}
    for sa in sentence_analyses:
        sent_text = sa['sentence_text']
        for kp in sa.get('keyphrase_analysis', sa.get('span_analysis', [])):
            cr = kp['coreference_analysis']
            if not cr.get('chain_found'): 
                continue
            cid = cr['chain_id']
            mention = kp.get('expanded_phrase', kp.get('phrase', kp.get('span')))
            clean = mention.strip().strip(string.punctuation).lower()
            if (len(clean)<2
                or clean in PRONOUNS
                or not any(c.isalpha() for c in clean)):
                continue
            entry = raw.setdefault(cid, {'mentions': set(), 'sentences': set()})
            entry['mentions'].add(clean)
            entry['sentences'].add(sent_text)

    clusters = []
    for cid, data in raw.items():
        if len(data['sentences']) < 2:
            continue
        clusters.append({
            'chain_id':  cid,
            'mentions':  sorted(data['mentions']),
            'sentences': list(data['sentences'])
        })
    return clusters


def build_cluster_graphs(clusters_dict):
    """
    Given clusters_dict: { chain_id: {'mentions':…, 'sentences':[…]}, … }
    returns { chain_id: networkx.Graph } where nodes are sentences,
    edges exist when cosine-sim > 0.1 in the SentenceTransformer embedding space.
    """
    graphs = {}
    for cid, data in clusters_dict.items():
        sents = data['sentences']
        if len(sents) < 2:
            continue
        emb = _model.encode(sents, convert_to_numpy=True)
        sim = cosine_similarity(emb)

        G = nx.Graph()
        for i, text in enumerate(sents):
            G.add_node(i, text=text)
        for i in range(len(sents)):
            for j in range(i+1, len(sents)):
                if sim[i, j] > 0.1:
                    G.add_edge(i, j, weight=float(sim[i, j]))
        graphs[cid] = G
    return graphs


def render_cluster_graph(cluster_id, clusters, cluster_graphs, focus_idx=None):
    """
    Display cluster graph using Plotly for interactivity
    """
    import plotly.graph_objects as go
    import networkx as nx
    
    G_full = cluster_graphs[cluster_id]
    info = next(c for c in clusters if c['chain_id'] == cluster_id)
    sentences, mentions = info['sentences'], info['mentions']
    
    G = G_full
    if focus_idx is not None and focus_idx in G_full:
        nodes = [focus_idx] + list(G_full.neighbors(focus_idx))
        G = G_full.subgraph(nodes)
    
    # Get layout
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        # Full sentence as hover text
        sent = sentences[node] if node < len(sentences) else f"Node {node}"
        # Truncate for display
        display_text = sent[:80] + "..." if len(sent) > 80 else sent
        node_text.append(f"[{node}] {display_text}")
        # Highlight focus node
        node_colors.append('red' if node == focus_idx else 'lightblue')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_text,
        text=[str(i) for i in G.nodes()],
        textposition="middle center",
        marker=dict(
            color=node_colors,
            size=20,
            line=dict(color='darkblue', width=2)
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f"Cluster {cluster_id} — mentions: {', '.join(mentions)}",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=500
                    ))
    
    fig.show()
    
    print("\nSentences in this cluster:")
    for idx, s in enumerate(sentences):
        tag = " ← FOCUS" if idx == focus_idx else ""
        print(f" [{idx}] {s[:100]}...{tag}")

def print_explanation_summary(pipeline_results):
    """Print summary statistics and consensus distribution"""
    counts = {}

    # Determine if this is keyphrase or span analysis
    sample_result = pipeline_results["results"][0] if pipeline_results["results"] else None
    is_keyphrase = sample_result and "keyphrase_analysis" in sample_result

    if is_keyphrase:
        keyphrase_counts = {"with_coreference": 0, "without_coreference": 0}

        for r in pipeline_results["results"]:
            c = r["consensus"]
            counts[c] = counts.get(c, 0) + 1

            # Count keyphrases with/without coreference
            for kp in r["keyphrase_analysis"]:
                if kp["coreference_analysis"]["chain_found"]:
                    keyphrase_counts["with_coreference"] += 1
                else:
                    keyphrase_counts["without_coreference"] += 1

        analysis_type = "Keyphrase"
        analysis_counts = keyphrase_counts

    else:
        span_counts = {"with_coreference": 0, "without_coreference": 0}

        for r in pipeline_results["results"]:
            c = r["consensus"]
            counts[c] = counts.get(c, 0) + 1

            # Count spans with/without coreference
            for span in r["span_analysis"]:
                if span["coreference_analysis"]["chain_found"]:
                    span_counts["with_coreference"] += 1
                else:
                    span_counts["without_coreference"] += 1

        analysis_type = "Span"
        analysis_counts = span_counts

    # Plot consensus distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Consensus distribution
    labels = list(counts.keys())
    vals = list(counts.values())
    ax1.bar(labels, vals)
    ax1.set_title("Classification Consensus Distribution")
    ax1.set_ylabel("Number of Sentences")
    ax1.tick_params(axis='x', rotation=45)

    # Analysis-coreference distribution
    analysis_labels = list(analysis_counts.keys())
    analysis_vals = list(analysis_counts.values())
    ax2.bar(analysis_labels, analysis_vals, color=['green', 'orange'])
    ax2.set_title(f"{analysis_type} Coreference Analysis")
    ax2.set_ylabel(f"Number of {analysis_type}s")
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    total_sentences = len(pipeline_results["results"])
    total_chains = len(pipeline_results["coref_chains"])

    print(f"\n📊 PIPELINE SUMMARY ({analysis_type} Analysis)")
    print("=" * 50)
    print(f"📄 Total sentences processed: {total_sentences}")
    print(f"🔗 Coreference chains found: {total_chains}")

    if is_keyphrase:
        kp_summary = pipeline_results["keyphrase_summary"]
        print(
            f"🔑 Sentences with keyphrases: {kp_summary['total_sentences_with_keyphrases']}")
        print(
            f"🔑 Total keyphrases extracted: {kp_summary['total_keyphrases']}")
        print(
            f"✅ Keyphrases with coreferences: {analysis_counts['with_coreference']}")
        print(
            f"❌ Keyphrases without coreferences: {analysis_counts['without_coreference']}")
    else:
        sp_summary = pipeline_results["span_summary"]
        print(
            f"🎯 Sentences with spans: {sp_summary['total_sentences_with_spans']}")
        print(f"🎯 Total spans extracted: {sp_summary['total_spans']}")
        print(
            f"✅ Spans with coreferences: {analysis_counts['with_coreference']}")
        print(
            f"❌ Spans without coreferences: {analysis_counts['without_coreference']}")


def analyze_and_display(idx, pipeline_results):
    """Display detailed analysis for a specific sentence"""
    r = pipeline_results["results"][idx]
    sent = r["sentence"]
    cons = r["consensus"]

    print(f"📝 Sentence [{idx}]: {sent}")
    print(f"🏷️  Consensus: {cons}")
    print(
        f"🤖 BERT: Label {r['primary_result']['label']}, Confidence {r['primary_result']['confidence']:.3f}")
    print(
        f"🔍 Similarity: Label {r['secondary_result']['label']}, Confidence {r['secondary_result']['confidence']:.3f}")
    print("-" * 80)

    # Check if this is keyphrase or span analysis
    if "keyphrase_analysis" in r:
        # Keyphrase analysis
        if r["keyphrase_analysis"]:
            print(f"🔑 KEYPHRASE ANALYSIS:")
            print("-" * 40)

            for i, kp in enumerate(r["keyphrase_analysis"], 1):
                print(f"\n🎯 Keyphrase {i}: '{kp['phrase']}'")

                if kp['expanded_phrase'] != kp['phrase']:
                    print(f"   🔄 Expanded: '{kp['expanded_phrase']}'")

                coref = kp["coreference_analysis"]
                if coref["chain_found"]:
                    print(f"   🔗 Coreference Chain Found!")
                    print(f"      Representative: {coref['representative']}")
                    print(
                        f"      Related mentions: {len(coref['related_mentions'])}")

                    if coref['related_mentions']:
                        print(f"      Examples:")
                        for mention in coref['related_mentions'][:3]:
                            print(f"         • '{mention['text']}'")
                else:
                    print(f"   ❌ No coreference chain found")
        else:
            print("🔑 No keyphrases extracted for this sentence")

    elif "span_analysis" in r:
        # Span analysis
        if r["span_analysis"]:
            print(f"🎯 SPAN ANALYSIS:")
            print("-" * 40)

            for i, span_info in enumerate(r["span_analysis"], 1):
                span = span_info["span"]
                print(f"\n🎯 Span {i}: '{span['text']}'")
                print(f"   📊 Importance: {span['importance']:.3f}")

                if span_info['expanded_phrase']['text'] != span['text']:
                    print(
                        f"   🔄 Expanded: '{span_info['expanded_phrase']['text']}'")

                coref = span_info["coreference_analysis"]
                if coref["chain_found"]:
                    print(f"   🔗 Coreference Chain Found!")
                    print(f"      Representative: {coref['representative']}")
                    print(
                        f"      Related mentions: {len(coref['related_mentions'])}")

                    if coref['related_mentions']:
                        print(f"      Examples:")
                        for mention in coref['related_mentions'][:3]:
                            print(f"         • '{mention['text']}'")
                else:
                    print(f"   ❌ No coreference chain found")
        else:
            print("🎯 No important spans found for this sentence")

    print()


def create_selector(pipeline_results):
    """Create interactive sentence selector widget"""
    opts = [("Select a sentence...", None)]

    # Determine analysis type
    sample_result = pipeline_results["results"][0] if pipeline_results["results"] else None
    is_keyphrase = sample_result and "keyphrase_analysis" in sample_result

    for i, r in enumerate(pipeline_results["results"]):
        txt = r["sentence"][:60].replace("\n", " ") + "..."
        cons = r['consensus']
        bert_conf = r['primary_result']['confidence']

        if is_keyphrase:
            num_items = len(r['keyphrase_analysis'])
            num_coref = sum(
                1 for kp in r['keyphrase_analysis'] if kp['coreference_analysis']['chain_found'])
            label = f"{i}: {cons} | BERT={bert_conf:.2f} | KP={num_items}({num_coref}) | {txt}"
        else:
            num_items = len(r['span_analysis'])
            num_coref = sum(
                1 for span in r['span_analysis'] if span['coreference_analysis']['chain_found'])
            label = f"{i}: {cons} | BERT={bert_conf:.2f} | SP={num_items}({num_coref}) | {txt}"

        opts.append((label, i))

    dd = widgets.Dropdown(
        options=opts,
        description='Sentence:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='1000px')
    )

    out = widgets.Output()

    def on_change(change):
        out.clear_output()
        if change["new"] is not None:
            with out:
                analyze_and_display(change["new"], pipeline_results)

    dd.observe(on_change, names="value")
    

    return dd, out
