from __future__ import annotations

import io
import json
import logging
import math
import re
import statistics
import urllib.parse
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader
from pypdf.errors import PdfReadWarning
from scrapling.fetchers import Fetcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .spec import DesignSpec


ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
PYPDF_LOGGER = logging.getLogger("pypdf")

TOPIC_KEYWORDS = {
    "network": ["network", "repeater", "entanglement", "spin-photon", "spin photon", "node", "quantum information"],
    "photonics": ["photonics", "waveguide", "resonator", "cavity", "grating", "disk resonator", "nanobeam", "purcell"],
    "sensing": ["magnetometry", "magnetometer", "imaging", "sensor", "sensing", "odmr"],
    "control": ["ramsey", "rabi", "hahn echo", "microwave", "coherent control", "hyperfine", "dressed state"],
    "fabrication": ["gds", "layout", "fabrication", "implant", "anneal", "gallium phosphide", "gap-on-diamond", "nanofabrication", "uniformity", "yield"],
}

APPLICATION_QUERY_MAP = {
    "quantum_repeater": [
        'all:"NV center" AND all:diamond AND (all:"quantum network" OR all:repeater OR all:entanglement)',
        'all:"NV center" AND all:diamond AND (all:photonic OR all:"spin-photon")',
        'all:"GaP-on-diamond" AND all:"NV center"',
    ],
    "memory": [
        'all:"NV center" AND all:diamond AND (all:memory OR all:entanglement)',
        'all:"NV center" AND all:diamond AND all:"quantum network"',
        'all:"NV center" AND all:diamond AND all:photonic',
    ],
    "processor": [
        'all:"NV center" AND all:diamond AND (all:processor OR all:control OR all:coupling)',
        'all:"NV center" AND all:diamond AND all:photonic',
    ],
    "magnetometer": [
        'all:"NV center" AND all:diamond AND (all:magnetometry OR all:sensing OR all:imaging)',
        'all:"NV center" AND all:diamond AND (all:odmr OR all:"hahn echo")',
        'all:"NV center" AND all:diamond AND all:control',
    ],
    "imager": [
        'all:"NV center" AND all:diamond AND (all:imaging OR all:sensing)',
        'all:"NV center" AND all:diamond AND all:photonic',
    ],
    "general": [
        'all:"NV center" AND all:diamond AND (all:photonic OR all:network OR all:sensing)',
        'all:"GaP-on-diamond" AND all:"NV center"',
    ],
}


@dataclass(slots=True)
class PaperRecord:
    url: str
    title: str
    abstract: str
    source: str
    published: str
    authors: list[str] = field(default_factory=list)
    pdf_url: str | None = None
    keywords: list[str] = field(default_factory=list)
    extracted_values: dict[str, list[float]] = field(default_factory=dict)
    relevance_score: float = 0.0
    pdf_excerpt: str = ""
    pdf_pages_scanned: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PaperKnowledge:
    application: str
    queries: list[str]
    query_terms: list[str]
    discovered_urls: list[str]
    papers: list[PaperRecord]
    topic_counts: dict[str, int]
    recommended_cavity_q: float
    recommended_waveguide_width_um: float
    recommended_t2_us: float
    recommended_magnetic_field_mT: float
    recommended_temp_k: float
    recommended_pitch_um: float
    recommended_optical_bus_ratio: float
    architecture_priors: dict[str, float]
    semantic_focus_terms: list[str]
    recommendation_provenance: dict[str, list[dict[str, Any]]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "application": self.application,
            "queries": self.queries,
            "query_terms": self.query_terms,
            "discovered_urls": self.discovered_urls,
            "papers": [paper.to_dict() for paper in self.papers],
            "topic_counts": self.topic_counts,
            "recommended_cavity_q": self.recommended_cavity_q,
            "recommended_waveguide_width_um": self.recommended_waveguide_width_um,
            "recommended_t2_us": self.recommended_t2_us,
            "recommended_magnetic_field_mT": self.recommended_magnetic_field_mT,
            "recommended_temp_k": self.recommended_temp_k,
            "recommended_pitch_um": self.recommended_pitch_um,
            "recommended_optical_bus_ratio": self.recommended_optical_bus_ratio,
            "architecture_priors": self.architecture_priors,
            "semantic_focus_terms": self.semantic_focus_terms,
            "recommendation_provenance": self.recommendation_provenance,
        }


def _query_arxiv(query: str, max_results: int) -> list[dict[str, Any]]:
    encoded = urllib.parse.urlencode({"search_query": query, "start": 0, "max_results": max_results})
    with urllib.request.urlopen(f"{ARXIV_API}?{encoded}", timeout=40) as response:
        payload = response.read().decode("utf-8", "ignore")
    root = ET.fromstring(payload)
    entries: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        links = entry.findall("atom:link", ARXIV_NS)
        abs_url = None
        pdf_url = None
        for link in links:
            href = link.attrib.get("href")
            rel = link.attrib.get("rel")
            if rel == "alternate":
                abs_url = href
            if link.attrib.get("title") == "pdf":
                pdf_url = href
        if not abs_url:
            continue
        authors = [author.findtext("atom:name", default="", namespaces=ARXIV_NS) for author in entry.findall("atom:author", ARXIV_NS)]
        entries.append(
            {
                "url": abs_url,
                "title": entry.findtext("atom:title", default="", namespaces=ARXIV_NS).strip(),
                "abstract": entry.findtext("atom:summary", default="", namespaces=ARXIV_NS).strip(),
                "published": entry.findtext("atom:published", default="", namespaces=ARXIV_NS),
                "authors": [author for author in authors if author],
                "pdf_url": pdf_url,
            }
        )
    return entries


def _first_nonempty(values: list[str | None]) -> str:
    for value in values:
        if value:
            stripped = value.strip()
            if stripped:
                return stripped
    return ""


def _extract_topic_hits(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {topic: sum(lowered.count(keyword) for keyword in keywords) for topic, keywords in TOPIC_KEYWORDS.items()}


def _extract_values(text: str) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {
        "cavity_q": [],
        "waveguide_width_um": [],
        "t2_us": [],
        "magnetic_field_mT": [],
        "temperature_k": [],
        "collection_efficiency": [],
        "purcell": [],
    }
    for match in re.finditer(r"\bQ(?:[- ]factor)?\b[^0-9]{0,24}([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)", text, re.IGNORECASE):
        try:
            q_value = float(match.group(1))
        except ValueError:
            continue
        if 1.0e3 <= q_value <= 1.0e8:
            values["cavity_q"].append(q_value)
    for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*(nm|um|µm|μm)\s*(?:wide\s+)?waveguide", text, re.IGNORECASE):
        magnitude = float(match.group(1))
        unit = match.group(2).lower().replace("μ", "u").replace("µ", "u")
        waveguide = magnitude / 1000.0 if unit == "nm" else magnitude
        if 0.05 <= waveguide <= 5.0:
            values["waveguide_width_um"].append(waveguide)
    for match in re.finditer(r"\bT2\b[^0-9]{0,18}([0-9]+(?:\.[0-9]+)?)\s*(ns|us|µs|μs|ms|s)\b", text, re.IGNORECASE):
        magnitude = float(match.group(1))
        unit = match.group(2).lower().replace("μ", "u").replace("µ", "u")
        scale = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}[unit]
        t2_value = magnitude * scale
        if 0.01 <= t2_value <= 1.0e8:
            values["t2_us"].append(t2_value)
    for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*(mT|T|G|gauss)\b", text, re.IGNORECASE):
        magnitude = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "t":
            magnitude *= 1000.0
        elif unit in {"g", "gauss"}:
            magnitude *= 0.1
        if 0.001 <= magnitude <= 5.0e4:
            values["magnetic_field_mT"].append(magnitude)
    for match in re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*K\b", text, re.IGNORECASE):
        temp_value = float(match.group(1))
        if 0.1 <= temp_value <= 500.0:
            values["temperature_k"].append(temp_value)
    for match in re.finditer(r"(?:efficiency|collection efficiency|external efficiency)[^0-9]{0,18}([0-9]+(?:\.[0-9]+)?)\s*%", text, re.IGNORECASE):
        values["collection_efficiency"].append(float(match.group(1)) / 100.0)
    for match in re.finditer(r"Purcell(?:\s+enhancement)?[^0-9]{0,18}([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE):
        values["purcell"].append(float(match.group(1)))
    return values


def _download_pdf_excerpt(pdf_url: str | None, max_pages: int = 4) -> tuple[str, int]:
    if not pdf_url:
        return "", 0
    try:
        with urllib.request.urlopen(pdf_url, timeout=50) as response:
            data = response.read()
        previous_level = PYPDF_LOGGER.level
        if previous_level == logging.NOTSET or previous_level < logging.ERROR:
            PYPDF_LOGGER.setLevel(logging.ERROR)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PdfReadWarning)
                reader = PdfReader(io.BytesIO(data), strict=False)
                pages = min(max_pages, len(reader.pages))
                excerpts = []
                for index in range(pages):
                    try:
                        text = reader.pages[index].extract_text() or ""
                    except Exception:
                        text = ""
                    if text:
                        excerpts.append(text)
                return "\n".join(excerpts), pages
        finally:
            PYPDF_LOGGER.setLevel(previous_level)
    except Exception:
        return "", 0


def _crawl_paper(entry: dict[str, Any]) -> PaperRecord:
    page = Fetcher.get(entry["url"])
    abstract_text_nodes = [text.strip() for text in page.css("blockquote.abstract::text").getall() if text and text.strip()]
    title = _first_nonempty(
        [
            page.css('meta[name="citation_title"]::attr(content)').get(),
            page.css('meta[property="og:title"]::attr(content)').get(),
            page.css("title::text").get(),
            entry.get("title"),
        ]
    )
    abstract = _first_nonempty(
        [
            " ".join(abstract_text_nodes),
            page.css('meta[name="description"]::attr(content)').get(),
            page.css('meta[property="og:description"]::attr(content)').get(),
            entry.get("abstract"),
        ]
    )
    published = _first_nonempty([page.css('meta[name="citation_date"]::attr(content)').get(), entry.get("published")])
    author_nodes = page.css('meta[name="citation_author"]::attr(content)').getall() or entry.get("authors", [])
    pdf_url = _first_nonempty([page.css('meta[name="citation_pdf_url"]::attr(content)').get(), entry.get("pdf_url")])
    pdf_excerpt, pages_scanned = _download_pdf_excerpt(pdf_url or None)
    combined_text = f"{title}\n{abstract}\n{pdf_excerpt}"
    topic_hits = _extract_topic_hits(combined_text)
    keywords = [topic for topic, count in topic_hits.items() if count > 0]
    return PaperRecord(
        url=entry["url"],
        title=title,
        abstract=abstract,
        source="arxiv",
        published=published,
        authors=author_nodes,
        pdf_url=pdf_url or None,
        keywords=keywords,
        extracted_values=_extract_values(combined_text),
        pdf_excerpt=pdf_excerpt[:4000],
        pdf_pages_scanned=pages_scanned,
    )


def _weighted_median(values: list[float], weights: list[float], fallback: float) -> float:
    clean = [(value, weight) for value, weight in zip(values, weights) if value > 0 and weight > 0]
    if not clean:
        return fallback
    clean.sort(key=lambda item: item[0])
    total = sum(weight for _, weight in clean)
    cursor = 0.0
    for value, weight in clean:
        cursor += weight
        if cursor >= total / 2.0:
            return value
    return clean[-1][0]


def _relevance_weights(spec: DesignSpec, papers: list[PaperRecord]) -> tuple[list[float], list[str]]:
    physicalized_terms = _physicalized_numeric_terms(spec)
    query_text = " ".join(
        [
            spec.design_name,
            spec.application,
            *physicalized_terms,
            "nv center spin qubit array",
            "photonic microwave cryogenic layout",
            spec.routing_preference,
        ]
    )
    corpus = [query_text] + [f"{paper.title}\n{paper.abstract}\n{paper.pdf_excerpt}" for paper in papers]
    if len(corpus) <= 1:
        return [], []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000)
    matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    max_score = float(scores.max()) if scores.size else 0.0
    normalized = [0.35 + 0.65 * (float(score) / max(max_score, 1e-9)) for score in scores]
    terms = vectorizer.get_feature_names_out()
    query_row = matrix[0].toarray().ravel()
    top_indices = np.argsort(query_row)[-8:][::-1] if query_row.size else []
    focus_terms = [terms[index] for index in top_indices if query_row[index] > 0]
    return normalized, focus_terms


def _filtered(values: list[float], spec_center: float, low_ratio: float, high_ratio: float, absolute_high: float | None = None) -> list[float]:
    low = spec_center * low_ratio
    high = spec_center * high_ratio
    if absolute_high is not None:
        high = min(high, absolute_high)
    return [value for value in values if low <= value <= high]


def _filtered_weighted(
    values: list[float],
    weights: list[float],
    spec_center: float,
    low_ratio: float,
    high_ratio: float,
    absolute_high: float | None = None,
) -> tuple[list[float], list[float]]:
    low = spec_center * low_ratio
    high = spec_center * high_ratio
    if absolute_high is not None:
        high = min(high, absolute_high)
    kept_values: list[float] = []
    kept_weights: list[float] = []
    for value, weight in zip(values, weights):
        if low <= value <= high:
            kept_values.append(value)
            kept_weights.append(weight)
    return kept_values, kept_weights


def _infer_architecture_priors(spec: DesignSpec, weighted_topics: dict[str, float]) -> dict[str, float]:
    base = spec.architecture_biases().copy()
    base["network_node"] += 0.03 * weighted_topics.get("network", 0.0) + 0.018 * weighted_topics.get("photonics", 0.0)
    base["hybrid_router"] += 0.022 * weighted_topics.get("photonics", 0.0) + 0.014 * weighted_topics.get("fabrication", 0.0) + 0.01 * weighted_topics.get("control", 0.0)
    base["sensor_dense"] += 0.026 * weighted_topics.get("sensing", 0.0) + 0.008 * weighted_topics.get("fabrication", 0.0)
    return base


def _physicalized_numeric_terms(spec: DesignSpec) -> list[str]:
    terms = ["NV-center spin qubit array", "diamond photonic control fabric"]
    if spec.target_qubits >= 48:
        terms.append("scalable multi-qubit array")
    if spec.target_qubits >= 96:
        terms.append("large-scale qubit array")
    if spec.target_logical_qubits >= 1:
        terms.append("fault-tolerant logical patch")
    if spec.target_logical_qubits >= 4:
        terms.append("multi-logical-qubit surface code fabric")
    if spec.max_latency_ns <= 220.0:
        terms.append("low-latency cryogenic control network")
    if spec.operating_temp_k <= 10.0:
        terms.append("cryogenic photonic interface")
    if spec.routing_preference == "low_loss":
        terms.append("low-loss resonator waveguide network")
    return list(dict.fromkeys(terms))


def _spec_query_terms(spec: DesignSpec) -> list[str]:
    terms = ["NV center diamond", "NV-center spin qubit array", *_physicalized_numeric_terms(spec)]
    if spec.application == "processor":
        terms.extend(["fault tolerant logical qubit", "cryogenic control fabric", "parallel readout"]) 
    if spec.target_logical_qubits > 1:
        terms.append("logical qubit decoder locality")
    if spec.target_t2_us >= 1000.0:
        terms.append("long spin coherence")
    return list(dict.fromkeys(terms))


def _build_application_queries(spec: DesignSpec) -> tuple[list[str], list[str]]:
    base_queries = list(APPLICATION_QUERY_MAP.get(spec.application.lower(), APPLICATION_QUERY_MAP["general"]))
    query_terms = _spec_query_terms(spec)
    extra_queries = [f'all:"NV center" AND all:diamond AND all:"{term}"' for term in query_terms[:4]]
    return list(dict.fromkeys(base_queries + extra_queries)), query_terms


def _recommendation_provenance(papers: list[PaperRecord], key: str, recommended_value: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for paper in papers:
        for value in paper.extracted_values.get(key, []):
            rows.append(
                {
                    "paper": paper.title or paper.url,
                    "url": paper.url,
                    "value": value,
                    "recommended_value": recommended_value,
                    "relevance_score": paper.relevance_score,
                    "distance": abs(value - recommended_value),
                }
            )
    rows.sort(key=lambda item: (item["distance"], -item["relevance_score"]))
    return rows[:4]


def _topic_recommendation_provenance(papers: list[PaperRecord], recommended_value: float, topic_keys: set[str], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for paper in papers:
        matched_topics = sorted(topic_keys & set(paper.keywords))
        if not matched_topics:
            continue
        rows.append(
            {
                "paper": paper.title or paper.url,
                "url": paper.url,
                "value": label,
                "recommended_value": recommended_value,
                "relevance_score": paper.relevance_score,
                "matched_topics": matched_topics,
                "distance": 0.0,
            }
        )
    rows.sort(key=lambda item: -item["relevance_score"])
    return rows[:4]


def build_paper_knowledge(spec: DesignSpec, max_papers: int = 6) -> PaperKnowledge:
    queries, query_terms = _build_application_queries(spec)
    discovered: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    per_query = max(2, max_papers // max(len(queries), 1) + 1)
    for query in queries:
        try:
            entries = _query_arxiv(query, max_results=per_query)
        except Exception:
            continue
        for entry in entries:
            if entry["url"] in seen_urls:
                continue
            discovered.append(entry)
            seen_urls.add(entry["url"])
            if len(discovered) >= max_papers:
                break
        if len(discovered) >= max_papers:
            break

    papers: list[PaperRecord] = []
    for entry in discovered[:max_papers]:
        try:
            papers.append(_crawl_paper(entry))
        except Exception:
            fallback_text = f"{entry.get('title', '')}\n{entry.get('abstract', '')}"
            papers.append(
                PaperRecord(
                    url=entry["url"],
                    title=entry.get("title", ""),
                    abstract=entry.get("abstract", ""),
                    source="arxiv_api",
                    published=entry.get("published", ""),
                    authors=entry.get("authors", []),
                    pdf_url=entry.get("pdf_url"),
                    keywords=[],
                    extracted_values=_extract_values(fallback_text),
                )
            )

    relevance_scores, focus_terms = _relevance_weights(spec, papers)
    for paper, score in zip(papers, relevance_scores):
        paper.relevance_score = score

    weighted_topics = {topic: 0.0 for topic in TOPIC_KEYWORDS}
    topic_counts = {topic: 0 for topic in TOPIC_KEYWORDS}
    cavity_q_values: list[float] = []
    cavity_q_weights: list[float] = []
    waveguide_values: list[float] = []
    waveguide_weights: list[float] = []
    t2_values: list[float] = []
    t2_weights: list[float] = []
    field_values: list[float] = []
    field_weights: list[float] = []
    temp_values: list[float] = []
    temp_weights: list[float] = []
    collection_values: list[float] = []
    collection_weights: list[float] = []

    for paper in papers:
        paper_text = f"{paper.title}\n{paper.abstract}\n{paper.pdf_excerpt}"
        hits = _extract_topic_hits(paper_text)
        for topic, count in hits.items():
            topic_counts[topic] += count
            weighted_topics[topic] += count * max(paper.relevance_score, 0.25)

        weight = max(paper.relevance_score, 0.25)
        for value in paper.extracted_values.get("cavity_q", []):
            cavity_q_values.append(value)
            cavity_q_weights.append(weight)
        for value in paper.extracted_values.get("waveguide_width_um", []):
            waveguide_values.append(value)
            waveguide_weights.append(weight)
        for value in paper.extracted_values.get("t2_us", []):
            t2_values.append(value)
            t2_weights.append(weight)
        for value in paper.extracted_values.get("magnetic_field_mT", []):
            field_values.append(value)
            field_weights.append(weight)
        for value in paper.extracted_values.get("temperature_k", []):
            temp_values.append(value)
            temp_weights.append(weight)
        for value in paper.extracted_values.get("collection_efficiency", []):
            collection_values.append(value)
            collection_weights.append(weight)

    filtered_fields, filtered_field_weights = _filtered_weighted(field_values, field_weights, spec.magnetic_field_mT, 0.25, 4.0, absolute_high=250.0)
    filtered_temps, filtered_temp_weights = _filtered_weighted(
        temp_values,
        temp_weights,
        spec.operating_temp_k,
        0.4,
        4.0,
        absolute_high=20.0 if spec.operating_temp_k <= 20.0 else 400.0,
    )
    filtered_t2, filtered_t2_weights = _filtered_weighted(t2_values, t2_weights, spec.target_t2_us, 0.1, 20.0)
    filtered_q, filtered_q_weights = _filtered_weighted(cavity_q_values, cavity_q_weights, max(spec.target_qubits * 1.0e4, 1.0e5), 0.1, 300.0, absolute_high=3.0e7)
    filtered_wg, filtered_wg_weights = _filtered_weighted(waveguide_values, waveguide_weights, 0.55, 0.25, 2.1, absolute_high=1.2)
    filtered_eff, filtered_eff_weights = _filtered_weighted(collection_values, collection_weights, 0.28, 0.1, 3.0, absolute_high=1.0)

    recommended_waveguide = min(0.80, max(0.36, _weighted_median(filtered_wg, filtered_wg_weights, 0.55))) if filtered_wg else 0.55
    recommended_cavity_q = max(1.0e5, _weighted_median(filtered_q, filtered_q_weights, 9.0e5)) if filtered_q else 9.0e5
    recommended_t2 = max(spec.target_t2_us, _weighted_median(filtered_t2, filtered_t2_weights, spec.target_t2_us)) if filtered_t2 else spec.target_t2_us
    recommended_field = _weighted_median(filtered_fields, filtered_field_weights, spec.magnetic_field_mT) if filtered_fields else spec.magnetic_field_mT
    recommended_temp = _weighted_median(filtered_temps, filtered_temp_weights, spec.operating_temp_k) if filtered_temps else spec.operating_temp_k
    recommended_eff = _weighted_median(filtered_eff, filtered_eff_weights, 0.28) if filtered_eff else 0.28

    photonics_bias = weighted_topics.get("photonics", 0.0)
    network_bias = weighted_topics.get("network", 0.0)
    sensing_bias = weighted_topics.get("sensing", 0.0)
    control_bias = weighted_topics.get("control", 0.0)
    recommended_pitch = min(52.0, max(24.0, 32.0 + 0.9 * network_bias + 0.6 * photonics_bias - 0.8 * sensing_bias + 0.35 * control_bias))
    recommended_bus_ratio = min(0.22, max(0.05, 0.055 + 0.007 * photonics_bias + 0.004 * network_bias + 0.04 * recommended_eff))

    return PaperKnowledge(
        application=spec.application,
        queries=queries,
        query_terms=query_terms,
        discovered_urls=[entry["url"] for entry in discovered],
        papers=papers,
        topic_counts=topic_counts,
        recommended_cavity_q=recommended_cavity_q,
        recommended_waveguide_width_um=recommended_waveguide,
        recommended_t2_us=recommended_t2,
        recommended_magnetic_field_mT=max(1.0, recommended_field),
        recommended_temp_k=max(1.0, recommended_temp),
        recommended_pitch_um=recommended_pitch,
        recommended_optical_bus_ratio=recommended_bus_ratio,
        architecture_priors=_infer_architecture_priors(spec, weighted_topics),
        semantic_focus_terms=focus_terms,
        recommendation_provenance={
            "cavity_q": _recommendation_provenance(papers, "cavity_q", recommended_cavity_q),
            "waveguide_width_um": _recommendation_provenance(papers, "waveguide_width_um", recommended_waveguide),
            "t2_us": _recommendation_provenance(papers, "t2_us", recommended_t2),
            "magnetic_field_mT": _recommendation_provenance(papers, "magnetic_field_mT", recommended_field),
            "temperature_k": _recommendation_provenance(papers, "temperature_k", recommended_temp),
            "pitch_um": _topic_recommendation_provenance(papers, recommended_pitch, {"photonics", "fabrication", "network"}, "pitch_prior"),
            "optical_bus_ratio": _topic_recommendation_provenance(papers, recommended_bus_ratio, {"photonics", "network"}, "bus_ratio_prior"),
        },
    )


def write_paper_artifacts(output_dir: Path, paper_knowledge: PaperKnowledge, include_markdown: bool = False) -> dict[str, str]:
    json_path = output_dir / "paper_knowledge.json"
    json_path.write_text(json.dumps(paper_knowledge.to_dict(), indent=2), encoding="utf-8")
    if not include_markdown:
        return {"paper_knowledge_json": str(json_path)}
    md_path = output_dir / "paper_digest.md"
    lines = [
        "# Paper Knowledge Digest",
        "",
        f"- Application: `{paper_knowledge.application}`",
        f"- Crawled papers: `{len(paper_knowledge.papers)}`",
        f"- Semantic focus: `{', '.join(paper_knowledge.semantic_focus_terms)}`",
        f"- Query terms: `{json.dumps(paper_knowledge.query_terms, ensure_ascii=False)}`",
        "",
        "## Aggregated Priors",
        "",
        f"- Recommended cavity Q: `{paper_knowledge.recommended_cavity_q:.1f}`",
        f"- Recommended waveguide width (um): `{paper_knowledge.recommended_waveguide_width_um:.3f}`",
        f"- Recommended T2 (us): `{paper_knowledge.recommended_t2_us:.2f}`",
        f"- Recommended magnetic field (mT): `{paper_knowledge.recommended_magnetic_field_mT:.2f}`",
        f"- Recommended temperature (K): `{paper_knowledge.recommended_temp_k:.2f}`",
        f"- Recommended pitch (um): `{paper_knowledge.recommended_pitch_um:.2f}`",
        f"- Recommended optical bus ratio: `{paper_knowledge.recommended_optical_bus_ratio:.3f}`",
        f"- Recommendation provenance: `{json.dumps(paper_knowledge.recommendation_provenance, ensure_ascii=False)}`",
        "",
        "## Papers",
        "",
    ]
    for paper in paper_knowledge.papers:
        lines.extend(
            [
                f"### {paper.title or paper.url}",
                f"- URL: `{paper.url}`",
                f"- Published: `{paper.published}`",
                f"- Relevance: `{paper.relevance_score:.3f}`",
                f"- Keywords: `{', '.join(paper.keywords)}`",
                f"- Abstract: {paper.abstract}",
                f"- PDF excerpt: {paper.pdf_excerpt[:900].replace(chr(10), ' ')}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"paper_knowledge_json": str(json_path), "paper_digest_md": str(md_path)}
