"""Import a garment from a product page URL.

Reads standard metadata (schema.org Product JSON-LD, Open Graph) and the
main product image. Only the public page and its image are fetched; no user
data is sent. Requests are SSRF-guarded: private, loopback and link-local
addresses are refused, redirects are re-validated, and sizes are capped.
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
from dataclasses import dataclass
from html import unescape
from urllib.parse import urljoin, urlparse

import httpx

MAX_HTML = 3 * 2**20
MAX_IMAGE = 15 * 2**20
UA = "Mozilla/5.0 (compatible; LookLabAI/0.1; +personal wardrobe import)"


class ImportErrorUser(ValueError):
    pass


@dataclass
class ProductInfo:
    url: str
    title: str | None
    image_url: str | None
    price: float | None
    currency: str | None
    brand: str | None


def _check_host(url: str) -> None:
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ImportErrorUser("only http(s) URLs are supported")
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80))
    except socket.gaierror as e:
        raise ImportErrorUser(f"cannot resolve {u.hostname}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise ImportErrorUser("refusing to fetch a private or local address")


def _get(url: str, limit: int, client: httpx.Client) -> tuple[bytes, str, str]:
    for _ in range(5):
        _check_host(url)
        with client.stream("GET", url, follow_redirects=False) as r:
            if r.is_redirect:
                url = urljoin(url, r.headers.get("location", ""))
                continue
            r.raise_for_status()
            buf = bytearray()
            for chunk in r.iter_bytes():
                buf += chunk
                if len(buf) > limit:
                    raise ImportErrorUser("response too large")
            return bytes(buf), r.headers.get("content-type", ""), str(r.url)
    raise ImportErrorUser("too many redirects")


def _meta(html: str, prop: str) -> str | None:
    for pat in (
        rf'<meta[^>]+(?:property|name)=["\']{re.escape(prop)}["\'][^>]*content=["\']([^"\']+)["\']',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]*(?:property|name)=["\']{re.escape(prop)}["\']',
    ):
        m = re.search(pat, html, re.I)
        if m:
            return unescape(m.group(1)).strip()
    return None


def _jsonld_products(html: str) -> list[dict]:
    out = []
    for m in re.finditer(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.S | re.I):
        try:
            data = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            d = stack.pop()
            if isinstance(d, dict):
                t = d.get("@type")
                if t == "Product" or (isinstance(t, list) and "Product" in t):
                    out.append(d)
                graph = d.get("@graph")
                if isinstance(graph, list):
                    stack.extend(graph)
    return out


def parse_product_html(html: str, base_url: str) -> ProductInfo:
    title = image = brand = currency = None
    price: float | None = None
    for p in _jsonld_products(html):
        title = title or p.get("name")
        img = p.get("image")
        if isinstance(img, list):
            img = img[0] if img else None
        if isinstance(img, dict):
            img = img.get("url")
        image = image or img
        b = p.get("brand")
        brand = brand or (b.get("name") if isinstance(b, dict) else b)
        offers = p.get("offers")
        if isinstance(offers, list):
            offers = offers[0] if offers else None
        if isinstance(offers, dict):
            pr = offers.get("price") or offers.get("lowPrice")
            if pr is not None and price is None:
                try:
                    price = float(str(pr).replace(",", "."))
                except ValueError:
                    pass
            currency = currency or offers.get("priceCurrency")
    title = title or _meta(html, "og:title")
    if not title:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
        title = unescape(m.group(1)).strip() if m else None
    image = image or _meta(html, "og:image") or _meta(html, "twitter:image")
    if price is None:
        pr = _meta(html, "product:price:amount") or _meta(html, "og:price:amount")
        if pr:
            try:
                price = float(pr.replace(",", "."))
            except ValueError:
                pass
    currency = currency or _meta(html, "product:price:currency") or _meta(html, "og:price:currency")
    return ProductInfo(
        url=base_url,
        title=title[:120] if title else None,
        image_url=urljoin(base_url, image) if image else None,
        price=price,
        currency=(currency or "").upper()[:3] or None,
        brand=str(brand)[:80] if brand else None,
    )


def fetch_product(url: str, transport: httpx.BaseTransport | None = None) -> tuple[ProductInfo, bytes]:
    with httpx.Client(timeout=15, headers={"User-Agent": UA}, transport=transport) as client:
        html_bytes, ctype, final_url = _get(url, MAX_HTML, client)
        if "html" not in ctype and not html_bytes.lstrip().startswith(b"<"):
            raise ImportErrorUser("URL is not a web page")
        info = parse_product_html(html_bytes.decode("utf-8", errors="replace"), final_url)
        if not info.image_url:
            raise ImportErrorUser("no product image found on that page; upload a photo instead")
        img, _, _ = _get(info.image_url, MAX_IMAGE, client)
        return info, img
