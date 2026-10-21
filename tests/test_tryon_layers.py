"""Layered try-on: pixel-exact paste-back, Qwen parsing, replaced-layer choice, graceful fallback."""

import asyncio
import os

import httpx
import numpy as np
import pytest
from PIL import Image

from looklab_ml.tryon import LocalFashnProvider, TryOnRequest
from looklab_ml.tryon.fashn_runtime import RunMetrics
from looklab_ml.tryon.layers import LayerError, LayerPlanner, _center_band, label_of, parse_items, recompose, select_replaced
from looklab_ml.vision.analyzers import ollama_chat


def test_recompose_keeps_pixels_outside_mask_identical():
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 255, (120, 90, 3), dtype=np.uint8)
    result = Image.fromarray(rng.integers(0, 255, (60, 45, 3), dtype=np.uint8))  # half size, like FASHN output
    edit = np.zeros((120, 90), bool)
    edit[30:80, 20:60] = True
    out, support = recompose(orig, result, edit)
    assert np.abs(out.astype(int) - orig.astype(int))[~support].max() == 0
    assert support.any() and (out[support] != orig[support]).any()


def test_parse_items_from_prefilled_and_truncated_output():
    text = '{"items": [{"name": "black jacket", "layer": "outer", "bbox_2d": [10, 20, 500, 600]}, {"name": "red sca'
    items = parse_items(text)
    assert [i["name"] for i in items] == ["black jacket"]
    assert parse_items("no json at all") == []
    assert parse_items('{"items": [{"name": "x", "bbox_2d": [1, 2, 3]}]}') == []  # box must have 4 numbers


def test_label_and_replaced_layer():
    assert [label_of(n) for n in ("puffer jacket", "wool scarf", "shoulder bag strap", "blue headband", "jeans", "ankle boots")] == [
        "top", "scarf", "bag", "hat", "pants", None]  # fmt: skip
    items = [
        {"name": "coat", "label": "top", "layer": "outer"},
        {"name": "shirt", "label": "top", "layer": "base"},
        {"name": "jeans", "label": "pants", "layer": "bottom"},
    ]
    assert [i["name"] for i in select_replaced(items, "outerwear")] == ["coat"]
    assert [i["name"] for i in select_replaced(items, "top")] == ["shirt"]  # outer layer kept when a base layer exists
    assert [i["name"] for i in select_replaced(items, "bottom")] == ["jeans"]
    assert len(select_replaced(items, "dress")) == 3
    assert [i["name"] for i in select_replaced(items[:1], "top")] == ["coat"]  # only an outer layer worn: top replaces it
    with pytest.raises(LayerError):
        select_replaced(items[:2], "bottom")


def test_ollama_chat_prefills_empty_think_block(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(json)
        return httpx.Response(200, json={"message": {"content": 'x"}]}'}}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    out = ollama_chat("http://ollama", "m", "p", "img", '{"items": [{"name": "', 5, keep_alive=0)
    assert sent["messages"][-1] == {"role": "assistant", "content": '<think>\n\n</think>\n\n{"items": [{"name": "'}
    assert sent["keep_alive"] == 0 and out == '{"items": [{"name": "x"}]}'


class _Runtime:
    parser_mode = "parser_free"

    def available(self):
        return True, "fake"

    def unload(self):
        pass

    def run(self, person, garment, category, photo_type, steps, scale, seed, n, seg_free, label_map):
        assert label_map is None  # fallback must not pass a label map
        return [person.resize((48, 64))], RunMetrics(1.0, None, None, "cpu", "float32")


def test_falls_back_to_segfree_when_ollama_is_absent(tmp_path):
    for n in ("p", "g"):
        Image.new("RGB", (96, 128), "gray").save(tmp_path / f"{n}.png")
    planner = LayerPlanner("http://127.0.0.1:9", "qwen3-vl:2b", str(tmp_path / "missing.pt"))
    req = TryOnRequest(str(tmp_path / "p.png"), str(tmp_path / "g.png"), "tops", garment_kind="outerwear", output_path=str(tmp_path / "o.png"))
    res = asyncio.run(LocalFashnProvider(_Runtime(), planner).generate(req))
    assert res.fidelity_mode == "segfree" and "ollama" in res.fidelity_reason and "SAM2" in res.fidelity_reason
    assert not res.uncovered_area and os.path.exists(res.output_image_path)


@pytest.mark.gpu
@pytest.mark.skipif(not os.environ.get("LOOKLAB_LAYERED_PERSON"), reason="set LOOKLAB_LAYERED_PERSON / LOOKLAB_LAYERED_GARMENT (photos stay local)")
def test_layered_real_photo(tmp_path):
    from looklab_ml.tryon import FashnRuntime

    rt = FashnRuntime(os.environ.get("LOOKLAB_FASHN_WEIGHTS_DIR", "weights/fashn-vton-1.5"))
    planner = LayerPlanner("http://127.0.0.1:11434", "qwen3-vl:2b", "weights/sam2/sam2.1_hiera_small.pt")
    req = TryOnRequest(
        os.environ["LOOKLAB_LAYERED_PERSON"], os.environ["LOOKLAB_LAYERED_GARMENT"], "tops",
        garment_kind="outerwear", output_path=str(tmp_path / "o.png"),
    )  # fmt: skip
    res = asyncio.run(LocalFashnProvider(rt, planner).generate(req))
    assert res.fidelity_mode == "layered", res.fidelity_reason
    assert "outside_mask_max_diff=0" in res.notes


def test_outerwear_needs_a_worn_outer_layer():
    with pytest.raises(LayerError):
        select_replaced([{"name": "tee", "label": "top", "layer": "base"}], "outerwear")


def test_outerwear_over_a_fresh_top_replaces_it():
    top = {"name": "tee", "label": "top", "layer": "base"}
    assert select_replaced([top], "outerwear", over_top=True) == [top]


def test_center_band_stays_in_the_upper_part_of_the_layer():
    m = np.zeros((100, 40), bool)
    m[:, 5:35] = True
    b = _center_band(m)
    assert b[:, 20].any() and not b[60:].any()  # a full-height band stretches the jacket (measured 2026-10-20)


def test_fidelity_lists_every_step():
    from types import SimpleNamespace as NS

    from looklab_api.jobs import _fidelity

    ok = NS(fidelity_mode="layered", fidelity_reason="a", stage_seconds={"qwen": 1.0}, uncovered_area=False, edit_fraction=0.1)
    bad = NS(fidelity_mode="segfree", fidelity_reason="b", stage_seconds={}, uncovered_area=False, edit_fraction=None)
    out = _fidelity([ok, bad], [{"garment_kind": "top"}, {"garment_kind": "outerwear"}])
    assert out["fidelity_mode"] == "segfree"
    assert [(s["garment_kind"], s["fidelity_mode"]) for s in out["fidelity_steps"]] == [("top", "layered"), ("outerwear", "segfree")]


def test_generate_does_not_block_the_event_loop(tmp_path):
    for n in ("p", "g"):
        Image.new("RGB", (96, 128), "gray").save(tmp_path / f"{n}.png")

    class Slow(_Runtime):
        def run(self, *a):
            import time

            time.sleep(0.4)  # stands in for FASHN
            return super().run(*a)

    req = TryOnRequest(str(tmp_path / "p.png"), str(tmp_path / "g.png"), "tops", output_path=str(tmp_path / "o.png"))

    async def main() -> float:
        gap, last, stop = 0.0, asyncio.get_running_loop().time(), False

        async def tick():
            nonlocal gap, last
            while not stop:
                await asyncio.sleep(0.01)
                now = asyncio.get_running_loop().time()
                gap, last = max(gap, now - last), now

        t = asyncio.create_task(tick())
        await LocalFashnProvider(Slow()).generate(req)
        stop = True
        await t
        return gap

    assert asyncio.run(main()) < 0.2
