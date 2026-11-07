"""Page 2: ask questions. Every answer shows its confidence, its claims, each citation with the result of the
six verification checks, and the cited page with the quote highlighted."""
from __future__ import annotations

import streamlit as st

from frontend import api, ui


def _evidence_viewer(result: dict, key: str) -> None:
    for req in ui.evidence_requests(result):
        if st.toggle(f"Show the cited page: {req['label']}", key=f"{key}-{req['key']}"):
            png, info = api.image(f"/documents/{req['document_id']}/pages/{req['page_number']}/image",
                                  quote=req["quote"], chunk_id=req["chunk_id"])
            if png is None:
                st.warning("The page image could not be loaded.")
                continue
            st.image(png, caption=f"{req['label']}   (yellow: quoted text, blue: clause region)")
            if not info.get("highlighted"):
                st.caption("The quote could not be located on the image (for example on a scanned page); "
                           "the clause region is outlined in blue.")


def _abstention(result: dict) -> None:
    a = result["abstention"]
    st.markdown(f"{ui.badge('INSUFFICIENT EVIDENCE')}  The documents do not contain enough evidence to answer this.")
    st.write(a["explanation"])
    for reason in a["reasons"]:
        st.caption(f"Why: {reason}")
    if a["related_passages"]:
        with st.expander("Related passages that were found (they do not answer the question)"):
            for p in a["related_passages"]:
                st.markdown(f"**{p['document']}, p.{p['page']}, clause {p['clause'] or '-'}**  (score {p['score']:.2f})")
                st.caption(p["preview"])
    if a.get("draft_rejected"):
        with st.expander("A draft answer was rejected by the verifier"):
            st.write(a["draft_rejected"])


def render_result(result: dict, key: str) -> None:
    if result.get("error"):
        st.error(result["error"])
        return
    if result["abstained"]:
        _abstention(result)
    else:
        st.markdown(f"**Confidence: {ui.badge(result['confidence'])}**" if result["use_retrieval"] else
                    ":gray[**No retrieval: this answer is ungrounded and has no citations**]")
        st.write(result["answer"])
        with st.expander("Why this confidence?"):
            for f in result["confidence_factors"]:
                st.markdown(f"- {f}")
            if result["reasoning_summary"]:
                st.caption(result["reasoning_summary"])
        for cmp in result["comparisons"]:
            st.info(f"**Comparison:** {cmp['result']}")
        for c in result["conflicts"]:
            (st.warning if c["classification"] == "CONFLICT" else st.info)(
                f"**{c['classification'].replace('_', ' ').title()}** - {c['explanation']}")
        if result["claims"]:
            st.markdown("##### Claims and citations")
            for i, claim in enumerate(result["claims"], 1):
                st.markdown(f"{i}. {ui.badge(claim['status'])}  {claim['claim']}")
                for ev in claim["evidence"]:
                    st.caption(f"{ui.badge(ev['status'])}  {ui.citation_label(ev['ref'])}  -  “{ev['ref']['supporting_text']}”")
                    st.caption(ui.check_line(ev["checks"]))
                    for n in ev["notes"]:
                        st.caption(f"note: {n}")
            _evidence_viewer(result, key)
    if result["retry_log"]:
        with st.expander("The verifier rejected a first draft"):
            for r in result["retry_log"]:
                st.caption(r)
    with st.expander("Retrieved passages"):
        for h in result["retrieved"]:
            c = h["chunk"]
            score = f"rerank {h['rerank_score']:.2f}" if h["rerank_score"] is not None else ""
            st.markdown(f"**{h['final_rank']}. {c['document_name']}, p.{c['page_number']}, clause {c['clause'] or '-'}**  {score}")
            st.caption(c["text"][:300])
    st.caption(ui.timing_caption(result))


def _ask(case_id: str, question: str, options: dict) -> dict | None:
    result = None
    with st.status("Working ...", expanded=True) as status:
        try:
            for ev in api.ask_stream(case_id, {"question": question, **options}):
                if ev["type"] == "status":
                    status.update(label=ev["text"])
                    st.write(ev["text"])
                elif ev["type"] == "result":
                    result = ev["result"]
                elif ev["type"] == "error":
                    status.update(label="Failed", state="error")
                    st.error(ev["text"])
                    return None
        except api.ApiError as e:
            status.update(label="Failed", state="error")
            st.error(str(e))
            return None
        status.update(label="Done", state="complete", expanded=False)
    return result


def render() -> None:
    st.title("Ask your documents")
    try:
        cases = api.get("/cases")
    except api.ApiError as e:
        st.error(str(e))
        return
    if not cases:
        st.info("Create a case and upload documents first (page: Cases and documents).")
        return
    ids = [c["id"] for c in cases]
    labels = {c["id"]: c["title"] for c in cases}
    current = st.session_state.get("case_id")
    case_id = st.sidebar.selectbox("Case", ids, index=ids.index(current) if current in ids else 0, format_func=labels.get)
    st.session_state["case_id"] = case_id
    st.sidebar.subheader("Retrieval")
    use_retrieval = st.sidebar.toggle("Use retrieval", True, help="Off = the same model WITHOUT the documents (baseline)")
    mode = st.sidebar.radio("Search mode", ["hybrid", "bm25", "dense"], disabled=not use_retrieval)
    rerank = st.sidebar.toggle("Cross-encoder reranking", True, disabled=not use_retrieval)
    k = st.sidebar.slider("Passages given to the model", 1, 10, 5, disabled=not use_retrieval)
    model = st.sidebar.text_input("Model (empty = configured)", "", placeholder="qwen3.5:9b")
    options = {"mode": mode, "rerank": rerank, "use_retrieval": use_retrieval, "final_k": k, "model": model.strip() or None}

    docs = api.get(f"/cases/{case_id}/documents")
    if not any(d["status"] == "indexed" for d in docs) and use_retrieval:
        st.warning("This case has no indexed documents yet.")
    history = st.session_state.setdefault("history", {}).setdefault(case_id, [])
    for i, (question, result) in enumerate(history):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            render_result(result, f"h{case_id}-{i}")
    if question := st.chat_input("Ask a question about the documents ..."):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            result = _ask(case_id, question, options)
            if result:
                render_result(result, f"h{case_id}-{len(history)}")
                history.append((question, result))
