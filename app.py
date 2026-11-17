"""Meeting Intelligence Assistant - Streamlit UI.  Run:  streamlit run app.py"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src import config
from src.embeddings import Embedder
from src.llm import PROVIDERS, LLMClient
from src.rag import MeetingRAG, location_label, hit_to_dict
from src.vector_store import VectorStore

st.set_page_config(page_title="Meeting Intelligence Assistant", page_icon="🗂️", layout="wide")

# ------------------------------------------------------------------------- styling
st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg,#6C5CE7 0%,#0984E3 100%); color:#fff; padding: 22px 28px;
       border-radius: 16px; margin-bottom: 18px;}
.hero h1 {margin:0; font-size: 1.8rem; color:#fff;} .hero p {margin:4px 0 0; opacity:.92;}
.metric {background:#F5F6FA; border-radius:14px; padding:14px 18px; border:1px solid #E6E8F0;}
.metric .v {font-size:1.7rem; font-weight:700; color:#4B3FCF;} .metric .l {font-size:.82rem; color:#667;}
.src {border:1px solid #E3E6F0; border-left:4px solid #6C5CE7; border-radius:10px; padding:9px 13px;
      margin:6px 0; background:#FAFAFF; font-size:.88rem;}
.src b {color:#3B2FB5;} .score {float:right; color:#0984E3; font-weight:600;}
.pill {display:inline-block; padding:2px 10px; border-radius:99px; font-size:.75rem; font-weight:600;}
.chip-note {color:#667; font-size:.85rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90 MB)…")
def get_store() -> VectorStore:
    return VectorStore(config.CHROMA_DIR, Embedder(config.EMBEDDING_MODEL))


try:
    store = get_store()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not start the vector store / embedding model: {e}")
    st.info("The first run must download the embedding model from huggingface.co (needs internet). "
            "Check your connection, or run:  python -c \"from sentence_transformers import SentenceTransformer as S; "
            "S('sentence-transformers/all-MiniLM-L6-v2')\"")
    st.stop()
if store.stale_reset and not st.session_state.get("stale_notice_shown"):
    st.session_state["stale_notice_shown"] = True
    st.warning("The saved vector index was built with a different embedding model, so it was reset. "
               "Please re-index your documents.", icon="♻️")
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("actions", {})
ss.setdefault("decisions", {})
ss.setdefault("summaries", {})

# ------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("LLM provider", list(PROVIDERS), format_func=lambda k: PROVIDERS[k]["label"])
    model = st.text_input("Model", PROVIDERS[provider]["model"], key=f"model_{provider}")
    api_key, base_url = "", ""
    if provider == "ollama":
        base_url = st.text_input("Ollama URL", "http://localhost:11434")
    else:
        api_key = st.text_input(f"{PROVIDERS[provider]['label']} API key", type="password",
                                help="Or set it in a .env file.")
    llm = LLMClient(provider, model, api_key, base_url)
    if st.button("Test LLM connection", width="stretch"):
        with st.spinner("Contacting the model…"):
            ok, msg = llm.test()
        (st.success if ok else st.error)(msg, icon="✅" if ok else "❌")
    elif llm.is_configured:
        st.success("LLM key set (click Test to verify)", icon="✅")
    else:
        st.warning("Add a Gemini API key (or choose Ollama) to enable answers.", icon="🔑")

    with st.expander("Retrieval settings"):
        top_k = st.slider("Chunks to retrieve (top-k)", 2, 15, config.DEFAULT_TOP_K)
        min_sim = st.slider("Min. similarity", 0.0, 0.8, config.DEFAULT_MIN_SIMILARITY, 0.05,
                            help="Chunks below this cosine similarity are ignored. Raise it for stricter "
                                 "'not found' behaviour.")

    st.divider()
    st.header("📤 Upload meetings")
    files = st.file_uploader("PDF, TXT, DOCX or MD", type=config.SUPPORTED_TYPES, accept_multiple_files=True)
    rag = MeetingRAG(store, llm)

    def ingest_many(items: list[tuple[str, bytes]]):
        bar = st.progress(0.0, text="Indexing…")
        log = []
        for i, (name, data) in enumerate(items, 1):
            try:
                (config.UPLOAD_DIR / name).write_bytes(data)
                n = rag.ingest(name, data)
                log.append(("ok", f"{name}: {n} chunks indexed"))
            except Exception as e:  # noqa: BLE001
                log.append(("err", f"{name}: indexing failed - {e}"))
            bar.progress(i / len(items), text=f"Indexed {i}/{len(items)}")
        bar.empty()
        ss["ingest_log"] = log  # survives the st.rerun() below so errors are never lost

    for kind, msg in ss.get("ingest_log", []):
        (st.success if kind == "ok" else st.error)(msg, icon="✅" if kind == "ok" else "❌")

    c1, c2 = st.columns(2)
    if c1.button("Index files", type="primary", disabled=not files, width="stretch"):
        ingest_many([(f.name, f.getvalue()) for f in files])
        st.rerun()
    if c2.button("Load samples", width="stretch", help="Index 3 demo meetings (2 TXT + 1 PDF)"):
        ingest_many([(p.name, p.read_bytes()) for p in sorted(config.SAMPLE_DIR.iterdir())
                     if p.suffix.lstrip(".") in config.SUPPORTED_TYPES])
        st.rerun()

docs = store.list_documents()
names = list(docs)

# ------------------------------------------------------------------------- header
st.markdown(
    '<div class="hero"><h1>🗂️ Meeting Intelligence Assistant</h1>'
    "<p>Ask questions across meetings, get summaries, and track decisions, action items, owners and deadlines "
    "- every answer comes with its source.</p></div>",
    unsafe_allow_html=True,
)
m1, m2, m3, m4 = st.columns(4)
for col, val, lab in [
    (m1, len(docs), "Meetings indexed"),
    (m2, sum(d["chunks"] for d in docs.values()), "Text chunks"),
    (m3, sum(len(v) for v in ss.actions.values()), "Action items found"),
    (m4, sum(len(v) for v in ss.decisions.values()), "Decisions found"),
]:
    col.markdown(f'<div class="metric"><div class="v">{val}</div><div class="l">{lab}</div></div>',
                 unsafe_allow_html=True)
st.write("")

if not docs:
    st.info(
        "**Get started:** upload meeting transcripts or minutes in the sidebar and click **Index files** - "
        "or click **Load samples** to try the demo meetings.",
        icon="👈",
    )
    st.stop()


# ------------------------------------------------------------------------- shared UI helpers
def render_sources(sources: list[dict], title: str = "📎 Sources"):
    with st.expander(f"{title} ({len(sources)})"):
        for s in sources:
            st.markdown(
                f'<div class="src"><span class="score">relevance {s["score"]:.2f}</span>'
                f'<b>[{s["label"]}]</b> {location_label(s)}</div>',
                unsafe_allow_html=True,
            )
            st.caption(s["text"])


def need_llm() -> bool:
    if not llm.is_configured:
        st.error("Configure an LLM in the sidebar first.", icon="🔑")
        return False
    return True


def loc(row_page, row_section) -> str:
    bits = []
    if row_page:
        bits.append(f"p.{row_page}")
    if row_section:
        bits.append(row_section)
    return " · ".join(bits) or "—"


# ------------------------------------------------------------------------- tabs
tab_ask, tab_sum, tab_act, tab_dec, tab_lib = st.tabs(
    ["💬 Ask", "📝 Summaries", "✅ Action items & deadlines", "🧭 Decisions", "📚 Library"]
)

# ---- Ask ----
with tab_ask:
    scope = st.multiselect("Search in (empty = all meetings)", names, placeholder="All meetings")
    st.markdown('<span class="chip-note">Try a quick question:</span>', unsafe_allow_html=True)
    quick = [
        "What decisions were made across all meetings?",
        "Why was the launch date changed?",
        "Who is responsible for the payment integration and by when?",
        "Which tasks are still pending?",
    ]
    cols = st.columns(len(quick))
    for c, q in zip(cols, quick):
        if c.button(q, width="stretch", key=f"q_{q}"):
            ss["pending_q"] = q

    for msg in ss.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])
            elif msg.get("related"):
                render_sources(msg["related"], "🔍 Closest (not relevant enough) excerpts")

    question = st.chat_input("Ask about discussions, decisions, owners, deadlines…") or ss.pop("pending_q", None)
    if question and need_llm():
        ss.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🤖"):
            ans = None
            with st.spinner("Searching meetings…"):
                try:
                    ans = rag.ask(question, scope or None, top_k, min_sim,
                                  [{"role": m["role"], "content": m["content"]} for m in ss.messages[:-1]])
                except Exception as e:  # noqa: BLE001
                    st.error(f"LLM error: {e}")
            if ans is None:
                ss.messages.pop()  # drop the unanswered question
            elif ans.found:
                (st.warning if ans.llm_error else st.markdown)(ans.text)
                render_sources(ans.sources)
            else:
                st.warning(ans.text, icon="🔎")
                if ans.related:
                    render_sources(ans.related, "🔍 Closest (not relevant enough) excerpts")
        if ans is not None:
            ss.messages.append({"role": "assistant", "content": ans.text if ans.found else f"⚠️ {ans.text}",
                                "sources": ans.sources, "related": ans.related})
    if ss.messages and st.button("🗑️ Clear chat"):
        ss.messages = []
        st.rerun()

# ---- Summaries ----
with tab_sum:
    sel = st.multiselect("Meetings to summarize", names, default=names[:1], key="sum_sel")
    force = st.checkbox("Regenerate (ignore cache)", key="sum_force")
    if st.button("Generate summaries", type="primary", key="sum_btn") and need_llm():
        for n in sel:
            with st.spinner(f"Summarizing {n}…"):
                try:
                    ss.summaries[n] = rag.summarize(n, force)
                except Exception as e:  # noqa: BLE001
                    st.error(f"{n}: {e}")
    for n in sel:
        if n in ss.summaries:
            with st.expander(f"📄 {n}", expanded=True):
                st.markdown(ss.summaries[n])
                st.download_button("Download .md", ss.summaries[n], f"{n}_summary.md", key=f"dl_{n}")

# ---- Action items ----
with tab_act:
    sel = st.multiselect("Meetings", names, default=names, key="act_sel")
    cA, cB = st.columns([1, 1])
    force = cB.checkbox("Re-extract (ignore cache)", key="act_force")
    if cA.button("Extract action items", type="primary", key="act_btn") and need_llm():
        bar = st.progress(0.0)
        for i, n in enumerate(sel, 1):
            try:
                ss.actions[n] = rag.extract(n, "actions", force)
            except Exception as e:  # noqa: BLE001
                st.error(f"{n}: {e}")
            bar.progress(i / len(sel), text=f"{n}")
        bar.empty()
        st.rerun()

    rows = [it for n in sel for it in ss.actions.get(n, [])]
    if not rows:
        st.caption("No action items yet - click **Extract action items**.")
    else:
        df = pd.DataFrame(rows)
        today = dt.date.today()
        df["due"] = pd.to_datetime(df["deadline_date"], errors="coerce")
        df["overdue"] = df["due"].notna() & (df["due"].dt.date < today) & (df["status"] != "Done")
        df["where"] = df.apply(lambda r: loc(r["page"], r["section"]), axis=1)

        f1, f2 = st.columns(2)
        owners = f1.multiselect("Owner", sorted(df.owner.unique()))
        stat = f2.multiselect("Status", sorted(df.status.unique()), default=[s for s in df.status.unique() if s != "Done"])
        if owners:
            df = df[df.owner.isin(owners)]
        if stat:
            df = df[df.status.isin(stat)]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Items", len(df))
        k2.metric("Pending / open", int((df.status != "Done").sum()))
        k3.metric("With deadline", int((df.deadline != "").sum()))
        k4.metric("Overdue", int(df.overdue.sum()))

        v_table, v_owner, v_dead = st.tabs(["Table", "By owner", "Deadlines"])
        show = df.rename(columns={"task": "Task", "owner": "Owner", "deadline": "Deadline", "due": "Due date",
                                  "status": "Status", "priority": "Priority", "meeting": "Meeting",
                                  "where": "Page / section", "evidence": "Evidence"})
        cols = ["Task", "Owner", "Deadline", "Due date", "Status", "Priority", "Meeting", "Page / section", "Evidence"]
        with v_table:
            st.dataframe(show[cols], width="stretch", hide_index=True,
                         column_config={"Task": st.column_config.TextColumn(width="large"),
                                        "Due date": st.column_config.DateColumn(format="DD MMM YYYY")})
            st.download_button("⬇️ Download CSV", show[cols].to_csv(index=False), "action_items.csv", "text/csv")
        with v_owner:
            for owner, g in df.groupby("owner"):
                open_n = int((g.status != "Done").sum())
                with st.expander(f"👤 {owner} - {len(g)} item(s), {open_n} open", expanded=True):
                    for _, r in g.iterrows():
                        due = f" - **due {r.deadline}**" if r.deadline else ""
                        flag = " ⚠️ overdue" if r.overdue else ""
                        st.markdown(f"- {r.task}{due} · _{r.status}_{flag}  \n"
                                    f"  <span class='chip-note'>📄 {r.meeting} · {r['where']}</span>",
                                    unsafe_allow_html=True)
        with v_dead:
            dd = df[df.deadline != ""].sort_values(["due", "deadline"], na_position="last")
            if dd.empty:
                st.caption("No deadlines mentioned.")
            for _, r in dd.iterrows():
                icon = "🔴" if r.overdue else ("🟢" if r.status == "Done" else "🟡")
                st.markdown(f"{icon} **{r.deadline}** - {r.task}  \n"
                            f"<span class='chip-note'>👤 {r.owner} · {r.status} · 📄 {r.meeting} · {r['where']}</span>",
                            unsafe_allow_html=True)

# ---- Decisions ----
with tab_dec:
    sel = st.multiselect("Meetings", names, default=names, key="dec_sel")
    force = st.checkbox("Re-extract (ignore cache)", key="dec_force")
    if st.button("Extract decisions", type="primary", key="dec_btn") and need_llm():
        for n in sel:
            with st.spinner(n):
                try:
                    ss.decisions[n] = rag.extract(n, "decisions", force)
                except Exception as e:  # noqa: BLE001
                    st.error(f"{n}: {e}")
        st.rerun()
    found = False
    for n in sel:
        for d in ss.decisions.get(n, []):
            found = True
            st.markdown(
                f'<div class="src"><b>{d["decision"]}</b><br>{d["context"]}<br>'
                f'<span class="chip-note">👥 {d["made_by"] or "—"} · 📄 {n} · {loc(d["page"], d["section"])}</span></div>',
                unsafe_allow_html=True)
    if not found:
        st.caption("No decisions yet - click **Extract decisions**.")

# ---- Library ----
with tab_lib:
    st.dataframe(
        pd.DataFrame([{"Document": n, "Type": d["file_type"].upper(), "Pages": str(d["pages"]) if d["pages"] else "—",
                       "Chunks": d["chunks"], "Indexed at": d["uploaded_at"]} for n, d in docs.items()]),
        width="stretch", hide_index=True)
    pick = st.selectbox("Inspect a document", names)
    with st.expander("View stored chunks"):
        for h in store.get_chunks(pick):
            st.markdown(f"**#{h.meta['chunk_index']}** · {location_label(hit_to_dict('', h))}")
            st.caption(h.text)
    with st.expander("🧬 Vector database (ChromaDB) details"):
        info = store.info()
        st.table(pd.DataFrame({"value": [str(v) for v in info.values()]}, index=list(info.keys())))
        chunk_no = st.number_input("Chunk # to inspect", 0, max(docs[pick]["chunks"] - 1, 0), 0)
        vec = store.get_vector(f"{pick}::{int(chunk_no)}")
        if vec:
            st.caption(f"Embedding for “{pick}” chunk {int(chunk_no)}: {len(vec)} floats, L2-normalised. First 8 values:")
            st.code(str([round(x, 4) for x in vec[:8]]))
    with st.expander("🛠️ Retrieval debugger (shows scores, ignores the threshold)"):
        dq = st.text_input("Test query", "Which payment vendor was chosen?")
        if dq:
            for h in store.search(dq, k=5, min_similarity=-1.0):
                st.markdown(f"**{h.score:.3f}** · {location_label(hit_to_dict('', h))}")
                st.caption(h.text[:300])
    d1, d2 = st.columns(2)
    if d1.button(f"🗑️ Delete “{pick}”"):
        store.delete_document(pick)
        for k in ("actions", "decisions", "summaries"):
            ss[k].pop(pick, None)
        st.rerun()
    if d2.button("⚠️ Reset entire index"):
        store.reset()
        for k in ("actions", "decisions", "summaries"):
            ss[k] = {}
        ss.messages = []
        st.rerun()
