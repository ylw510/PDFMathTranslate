"""Sentence boxes recorded while typesetting, in preview coordinates."""

from __future__ import annotations


def is_sentence_end(chars: list[dict], index: int) -> bool:
    """Same cut as the preview: 。！？ always, .!? before space or an uppercase letter."""
    glyph = chars[index]
    if glyph.get("formula"):
        return False
    text = glyph.get("c") or ""
    if text in "。！？":
        return True
    if text not in ".!?":
        return False
    nxt = ""
    for later in chars[index + 1 :]:
        if later.get("formula"):
            continue
        nxt = later.get("c") or ""
        break
    if nxt == "" or nxt.isspace() or nxt in "\"'”’）)":
        return True
    return nxt.isupper()


def _line_rects(glyphs: list[dict], page_w: float, page_h: float) -> list[list[float]]:
    """PDF boxes (y up) -> normalized [x, y, w, h] with y down, one rect per line."""
    visible = [
        glyph
        for glyph in glyphs
        if glyph.get("bbox") and not (glyph.get("c") or "").isspace()
    ]
    groups: list[list[dict]] = []
    for glyph in visible:
        y0 = glyph["bbox"][1]
        if groups and abs(y0 - groups[-1][0]["bbox"][1]) < 2:
            groups[-1].append(glyph)
        else:
            groups.append([glyph])
    rects = []
    for group in groups:
        boxes = [glyph["bbox"] for glyph in group]
        x0 = min(box[0] for box in boxes) + 0.3
        y0 = min(box[1] for box in boxes) + 0.4
        x1 = max(box[2] for box in boxes) - 0.3
        y1 = max(box[3] for box in boxes) - 0.4
        if x1 <= x0 or y1 <= y0 or page_w <= 0 or page_h <= 0:
            continue
        rects.append(
            [
                round(x0 / page_w, 4),
                round((page_h - y1) / page_h, 4),
                round((x1 - x0) / page_w, 4),
                round((y1 - y0) / page_h, 4),
            ]
        )
    return rects


def sentence_units(
    glyphs: list[dict], page_w: float, page_h: float
) -> list[list[list[float]]]:
    """Split a glyph stream into sentences and return each sentence's line rects."""
    units: list[list[list[float]]] = []
    current: list[dict] = []
    for index, glyph in enumerate(glyphs):
        current.append(glyph)
        if is_sentence_end(glyphs, index):
            rects = _line_rects(current, page_w, page_h)
            if rects:
                units.append(rects)
            current = []
    if current:
        rects = _line_rects(current, page_w, page_h)
        if rects:
            units.append(rects)
    return units


def _centers(units: list[list[list[float]]]) -> list[float]:
    return [sum(rect[1] + rect[3] / 2 for rect in unit) / len(unit) for unit in units]


def _monotone_align(src_pos: list[float], dst_pos: list[float]) -> list[tuple[int, int]]:
    """Order-preserving alignment. One side may merge, neither side may cross."""
    src_n, dst_n = len(src_pos), len(dst_pos)
    inf = 10**9
    cost = [[inf] * dst_n for _ in range(src_n)]
    back = [[None] * dst_n for _ in range(src_n)]
    cost[0][0] = abs(src_pos[0] - dst_pos[0])
    for src_i in range(1, src_n):
        cost[src_i][0] = cost[src_i - 1][0] + abs(src_pos[src_i] - dst_pos[0])
        back[src_i][0] = (src_i - 1, 0)
    for dst_i in range(1, dst_n):
        cost[0][dst_i] = cost[0][dst_i - 1] + abs(src_pos[0] - dst_pos[dst_i])
        back[0][dst_i] = (0, dst_i - 1)
    for src_i in range(1, src_n):
        for dst_i in range(1, dst_n):
            choices = (
                (cost[src_i - 1][dst_i], (src_i - 1, dst_i)),
                (cost[src_i][dst_i - 1], (src_i, dst_i - 1)),
                (cost[src_i - 1][dst_i - 1], (src_i - 1, dst_i - 1)),
            )
            best, origin = min(choices, key=lambda item: item[0])
            cost[src_i][dst_i] = best + abs(src_pos[src_i] - dst_pos[dst_i])
            back[src_i][dst_i] = origin
    path = []
    cursor = (src_n - 1, dst_n - 1)
    while cursor is not None:
        path.append(cursor)
        cursor = back[cursor[0]][cursor[1]]
    path.reverse()
    return path


def pair_sentence_units(
    src_units: list[list[list[float]]], dst_units: list[list[list[float]]]
) -> list[tuple[list[list[float]], list[list[float]]]]:
    """Zip equal counts. Unequal counts stay in reading order, so a merged sentence is shared."""
    if not src_units or not dst_units:
        return []
    if len(src_units) == len(dst_units):
        return list(zip(src_units, dst_units))

    grouped: dict[int, list[int]] = {}
    for src_i, dst_i in _monotone_align(_centers(src_units), _centers(dst_units)):
        bucket = grouped.setdefault(src_i, [])
        if dst_i not in bucket:
            bucket.append(dst_i)
    return [
        (
            src_units[src_i],
            [rect for dst_i in dst_ids for rect in dst_units[dst_i]],
        )
        for src_i, dst_ids in grouped.items()
    ]
