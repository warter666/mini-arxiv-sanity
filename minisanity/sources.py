"""Paper sources: an offline fixture and the real arXiv Atom API.

Security posture of the network source: pinned https endpoint on
export.arxiv.org, DNS resolution checked against public unicast addresses
(blocks loopback/private/link-local/metadata ranges, mitigating DNS
rebinding), redirects disabled, response size capped, and DTD/entity
declarations rejected before XML parsing.
"""

import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from .store import Paper

ARXIV_HOST = "export.arxiv.org"
ARXIV_API = f"https://{ARXIV_HOST}/api/query"
ARXIV_PORT = 443
MAX_RESPONSE_BYTES = 20 * 1024 * 1024
ATOM = "{http://www.w3.org/2005/Atom}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError(f"redirect to {newurl} refused")


class FixtureSource:
    """Deterministic offline papers for tests and demos."""

    def __init__(self, papers=None):
        if papers is None:
            papers = [
                Paper("2401.00001", "Scaling Laws for Neural Language Models",
                      "We study how loss scales with compute, data and parameters for "
                      "transformer language models trained on text corpora.",
                      ["Kaplan"], ["cs.CL"]),
                Paper("2401.00002", "Attention Is All You Need",
                      "We propose the Transformer, a network architecture based solely "
                      "on attention mechanisms, achieving state-of-the-art results on "
                      "language modeling and machine translation.",
                      ["Vaswani"], ["cs.CL"]),
                Paper("2401.00003", "Deep Residual Learning for Image Recognition",
                      "We present residual learning frameworks that ease the training "
                      "of deep convolutional networks for visual recognition tasks.",
                      ["He"], ["cs.CV"]),
                Paper("2401.00004", "ImageNet Classification with Deep CNNs",
                      "A large convolutional neural network trained on ImageNet "
                      "achieves record-breaking results in visual object recognition.",
                      ["Krizhevsky"], ["cs.CV"]),
                Paper("2401.00005", "Training Language Models with Reinforcement "
                      "Learning from Human Feedback",
                      "We align language model behavior with human preferences using "
                      "reinforcement learning from human feedback reward models.",
                      ["Ouyang"], ["cs.CL", "cs.LG"]),
            ]
        self.papers = papers

    def fetch(self, max_results=100, query=None):
        return self.papers[:max_results]


class ArxivSource:
    def __init__(self, categories=("cs.CL", "cs.LG", "cs.CV")):
        self.categories = list(categories)

    def _build_url(self, max_results, query):
        q = query or " OR ".join(f"cat:{c}" for c in self.categories)
        params = urllib.parse.urlencode(
            {"search_query": q, "sortBy": "submittedDate",
             "sortOrder": "descending", "max_results": int(max_results)})
        return f"{ARXIV_API}?{params}"

    @staticmethod
    def _check_dns():
        """Resolve the pinned host and refuse non-public addresses."""
        infos = socket.getaddrinfo(ARXIV_HOST, ARXIV_PORT, proto=socket.IPPROTO_TCP)
        addrs = {info[4][0] for info in infos}
        for addr in addrs:
            ip = ipaddress.ip_address(addr)
            if not ip.is_global:
                raise ValueError(f"{ARXIV_HOST} resolved to non-public address {addr}, refusing")

    def _fetch_bytes(self, url):
        if urllib.parse.urlparse(url).hostname != ARXIV_HOST:
            raise ValueError("URL host drifted from the pinned arXiv endpoint")
        self._check_dns()
        opener = urllib.request.build_opener(_NoRedirect)
        req = urllib.request.Request(url, headers={"User-Agent": "minisanity/0.1"})
        with opener.open(req, timeout=60) as resp:  # export.arxiv.org can be slow
            data = resp.read(MAX_RESPONSE_BYTES + 1)
        if len(data) > MAX_RESPONSE_BYTES:
            raise ValueError("arXiv response exceeds size cap")
        return data

    @staticmethod
    def _safe_xml(data: bytes) -> ET.Element:
        if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
            raise ValueError("arXiv response contains DTD/entity declarations, rejecting")
        return ET.fromstring(data)

    def fetch(self, max_results=100, query=None):
        url = self._build_url(max_results, query)
        root = self._safe_xml(self._fetch_bytes(url))
        papers = []
        for entry in root.findall(f"{ATOM}entry"):
            arxiv_id = entry.find(f"{ATOM}id").text.split("/abs/")[-1]
            title = re.sub(r"\s+", " ", entry.find(f"{ATOM}title").text).strip()
            abstract = re.sub(r"\s+", " ", entry.find(f"{ATOM}summary").text).strip()
            authors = [a.find(f"{ATOM}name").text for a in entry.findall(f"{ATOM}author")]
            papers.append(Paper(arxiv_id, title, abstract, authors, list(self.categories)))
        return papers
