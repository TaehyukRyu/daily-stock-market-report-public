"""README용 SVG 3장 생성 — diagram-design 기본 스킨, 다크 모드 내장.

실행: python doc/make_readme_svgs.py assets   (저장소 루트에서, assets/*.svg를 덮어쓴다)
"""
import math
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

SANS = "'Pretendard','Apple SD Gothic Neo','Malgun Gothic','Noto Sans KR',sans-serif"
MONO = "ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace"

STYLE = f"""<style>
.bg{{fill:#f5f5f5}} .mask{{fill:#f5f5f5}}
.card{{fill:#ffffff;stroke:#2d3142}}
.inp{{fill:rgba(79,93,117,0.10);stroke:#7a8399}}
.opt{{fill:rgba(45,49,66,0.02);stroke:rgba(45,49,66,0.40);stroke-dasharray:4 3}}
.ext{{fill:rgba(45,49,66,0.03);stroke:rgba(45,49,66,0.30)}}
.store{{fill:rgba(45,49,66,0.05);stroke:#4f5d75}}
.focal{{fill:rgba(235,108,54,0.08);stroke:#eb6c36}}
.hub{{fill:#2d3142}}
.ln{{stroke:#4f5d75;fill:none}} .ln-soft{{stroke:#7a8399;fill:none}} .ln-acc{{stroke:#eb6c36;fill:none}}
.m{{fill:#4f5d75}} .m-soft{{fill:#7a8399}} .m-acc{{fill:#eb6c36}}
.rule{{stroke:rgba(45,49,66,0.12)}}
.t-ink{{fill:#2d3142}} .t-muted{{fill:#4f5d75}} .t-soft{{fill:#7a8399}} .t-acc{{fill:#d9591f}} .t-hub{{fill:#f5f5f5}}
.dot{{fill:#2d3142}}
.sans{{font-family:{SANS}}} .mono{{font-family:{MONO}}}
@media (prefers-color-scheme: dark){{
.bg{{fill:#2d3142}} .mask{{fill:#2d3142}}
.card{{fill:#393e53;stroke:#f5f5f5}}
.inp{{fill:rgba(191,192,192,0.10);stroke:#8e98ac}}
.opt{{fill:rgba(245,245,245,0.03);stroke:rgba(245,245,245,0.45)}}
.ext{{fill:rgba(245,245,245,0.04);stroke:rgba(245,245,245,0.35)}}
.store{{fill:rgba(245,245,245,0.06);stroke:#bfc0c0}}
.focal{{fill:rgba(240,138,89,0.10);stroke:#f08a59}}
.hub{{fill:#f5f5f5}}
.ln{{stroke:#bfc0c0}} .ln-soft{{stroke:#8e98ac}} .ln-acc{{stroke:#f08a59}}
.m{{fill:#bfc0c0}} .m-soft{{fill:#8e98ac}} .m-acc{{fill:#f08a59}}
.rule{{stroke:rgba(245,245,245,0.12)}}
.t-ink{{fill:#f5f5f5}} .t-muted{{fill:#bfc0c0}} .t-soft{{fill:#8e98ac}} .t-acc{{fill:#f08a59}} .t-hub{{fill:#2d3142}}
.dot{{fill:#f5f5f5}}
}}
</style>"""

MARKERS = """<marker id="{p}-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon class="m" points="0 0, 8 3, 0 6"/></marker>
<marker id="{p}-arrow-soft" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon class="m-soft" points="0 0, 8 3, 0 6"/></marker>
<marker id="{p}-arrow-acc" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><polygon class="m-acc" points="0 0, 8 3, 0 6"/></marker>"""


def head(slug, w, h, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-labelledby="{slug}-title {slug}-desc">\n'
            f'<title id="{slug}-title">{title}</title>\n<desc id="{slug}-desc">{desc}</desc>\n'
            f'<defs>\n{STYLE}\n{MARKERS.format(p=slug)}\n</defs>\n'
            f'<rect class="bg" width="{w}" height="{h}" rx="8"/>\n')


def arrow(slug, d, kind="", dashed=False):
    cls = {"": "ln", "soft": "ln-soft", "acc": "ln-acc"}[kind]
    mk = f"{slug}-arrow" + (f"-{kind}" if kind else "")
    dash = ' stroke-dasharray="5,4"' if dashed else ""
    sw = "1" if dashed else "1.2"
    return f'<path class="{cls}" d="{d}" stroke-width="{sw}"{dash} marker-end="url(#{mk})"/>\n'


def label(x, y, w, text, cls="t-soft", anchor="middle", weight="500"):
    """마스크 사각형(16px 높이) + 12px 한글 라벨. (x,y) = 마스크 왼쪽 위."""
    tx = x + w / 2 if anchor == "middle" else x + 4
    return (f'<rect class="mask" x="{x}" y="{y}" width="{w}" height="16" rx="2"/>\n'
            f'<text class="sans {cls}" x="{tx:g}" y="{y + 12}" font-size="12" font-weight="{weight}" '
            f'text-anchor="{anchor}">{text}</text>\n')


def node(x, y, w, h, cls, name, sub, sub_mono=False, name_cls="t-ink"):
    cx = x + w / 2
    out = ""
    if cls not in ("card",):
        out += f'<rect class="mask" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>\n'
    out += f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6" stroke-width="1"/>\n'
    out += (f'<text class="sans {name_cls}" x="{cx:g}" y="{y + h / 2 - 2:g}" font-size="16" font-weight="600" '
            f'text-anchor="middle">{name}</text>\n')
    fam = "mono" if sub_mono else "sans"
    out += (f'<text class="{fam} t-muted" x="{cx:g}" y="{y + h / 2 + 18:g}" font-size="12" '
            f'text-anchor="middle">{sub}</text>\n')
    return out


def eyebrow(x, y, text):
    return (f'<text class="mono t-soft" x="{x}" y="{y}" font-size="12" letter-spacing="0.08em">{text}</text>\n')


def legend(slug, y, x0, x1, items):
    out = f'<line class="rule" x1="{x0}" y1="{y - 24}" x2="{x1}" y2="{y - 24}" stroke-width="0.8"/>\n'
    x = x0
    for kind, text in items:
        if kind == "solid":
            out += f'<line class="ln" x1="{x}" y1="{y - 4}" x2="{x + 28}" y2="{y - 4}" stroke-width="1.2"/>\n'
            tx = x + 36
        elif kind == "dashed":
            out += (f'<line class="ln-soft" x1="{x}" y1="{y - 4}" x2="{x + 28}" y2="{y - 4}" stroke-width="1" '
                    f'stroke-dasharray="5,4"/>\n')
            tx = x + 36
        elif kind == "focal":
            out += f'<rect class="focal" x="{x}" y="{y - 12}" width="20" height="14" rx="3" stroke-width="1"/>\n'
            tx = x + 28
        else:  # hub
            out += f'<rect class="hub" x="{x}" y="{y - 12}" width="20" height="14" rx="3"/>\n'
            tx = x + 28
        out += f'<text class="sans t-muted" x="{tx}" y="{y}" font-size="12">{text}</text>\n'
        # 다음 항목 위치: 글자 폭(한글 1em, 그 외 0.6em) + 여백
        tw = sum(12 if ord(c) > 0x2000 else 7.2 for c in text)
        x = int(math.ceil((tx + tw + 32) / 4) * 4)
    return out


# ───────────── ① 하루 흐름 ─────────────
def daily_flow():
    s, W, H = "daily-flow", 944, 320
    xs = [32, 184, 336, 488, 640, 792]      # 폭 120, 간격 32
    y, h, w = 160, 64, 120
    cy = y + h // 2
    o = head(s, W, H, "하루 흐름",
             "유니버스 110종목을 뉴스·공시로 스크리닝하고, 투표자 8명의 표를 파이썬 규칙이 방향으로 확정한 뒤 "
             "chief가 근거를 서술해 Notion에 발행하는 흐름. 토론은 의견이 팽팽할 때만 열린다.")
    o += eyebrow(32, 36, "EVERY WEEKDAY · GITHUB ACTIONS")
    # 화살표 먼저
    for a, b in zip(xs, xs[1:]):
        o += arrow(s, f"M{a + w},{cy} H{b - 2}")
    o += arrow(s, "M396,160 V96 Q396,88 404,88 H486", "soft", dashed=True)     # 투표 → 토론
    o += arrow(s, "M608,88 H692 Q700,88 700,96 V158", "soft", dashed=True)     # 토론 → chief
    o += '<line class="ln-soft" x1="548" y1="228" x2="548" y2="240" stroke-width="1" stroke-dasharray="2,3"/>\n'
    # 노드
    o += node(488, 56, 120, 64, "opt", "토론", "의견이 팽팽할 때")
    specs = [("inp", "유니버스", "110종목", False), ("card", "스크리닝", "뉴스 급증·공시", False),
             ("card", "투표 8명", "LLM 7 + 규칙 1", False), ("focal", "방향 결정", "decide()", True),
             ("card", "chief 서술", "근거·리스크", False), ("ext", "Notion", "하루 1건 발행", False)]
    for x, (cls, name, sub, mono) in zip(xs, specs):
        o += node(x, y, w, h, cls, name, sub, mono)
    o += ('<text class="sans t-muted" x="548" y="256" font-size="12" text-anchor="middle">'
          '표가 같으면 방향도 같다</text>\n')
    o += legend(s, 304, 32, 912, [("solid", "매일"), ("dashed", "조건부"), ("focal", "LLM이 아닌 코드")])
    return s, o + "</svg>\n"


# ───────────── ② 채점·학습 루프 ─────────────
def feedback_loop():
    s, W, H = "feedback-loop", 688, 664
    C, R, N = (344, 332), 240, 6
    sw, sh, hw, hh = 160, 64, 200, 104
    stations = [("예측 기록", "D일 · 8명 + chief", True),
                ("D+1 복기", "교훈 2~4문장", True),
                ("D+5 채점", "실현 수익 기준", True),
                ("증거 게이트", "30표본 + z검정", False),
                ("가중치 갱신", "통과할 때만", True),
                ("다음 판단", "교훈·가중치 주입", False)]
    focal = 3
    centers = []
    for k in range(N):
        th = math.radians(-90 + k * 360 / N)
        centers.append((round((C[0] + R * math.cos(th)) / 4) * 4, round((C[1] + R * math.sin(th)) / 4) * 4))

    def ang(p):
        return math.atan2(p[1] - C[1], p[0] - C[0])

    def hits(cx, cy):
        pts = []
        x0, x1, y0, y1 = cx - sw / 2, cx + sw / 2, cy - sh / 2, cy + sh / 2
        for xe in (x0, x1):
            d = R * R - (xe - C[0]) ** 2
            if d >= 0:
                for sg in (1, -1):
                    yy = C[1] + sg * math.sqrt(d)
                    if y0 <= yy <= y1:
                        pts.append((xe, yy))
        for ye in (y0, y1):
            d = R * R - (ye - C[1]) ** 2
            if d >= 0:
                for sg in (1, -1):
                    xx = C[0] + sg * math.sqrt(d)
                    if x0 <= xx <= x1:
                        pts.append((xx, ye))
        return pts

    two_pi = 2 * math.pi
    exits, entries = [], []
    for k, (cx, cy) in enumerate(centers):
        th = ang((cx, cy))
        pts = hits(cx, cy)
        exits.append(min(pts, key=lambda p: (ang(p) - th) % two_pi))
        entries.append(min(pts, key=lambda p: (th - ang(p)) % two_pi))

    o = head(s, W, H, "채점·학습 루프",
             "매일의 예측을 기록하고 D+1에 복기 교훈을, D+5에 실현 수익 채점을 남긴 뒤, 증거 게이트를 통과할 때만 "
             "가중치를 갱신해 다음 판단에 주입하는 순환. 모든 기록은 mentions.db 한 곳에 쌓인다.")
    o += eyebrow(56, 32, "AFTER MARKET CLOSE")
    # 링 화살표
    for k in range(N):
        j = (k + 1) % N
        ex = exits[k]
        phi = ang(entries[j]) - 1.2 / R
        q = (C[0] + R * math.cos(phi), C[1] + R * math.sin(phi))
        o += arrow(s, f"M{ex[0]:.3f},{ex[1]:.3f} A{R},{R} 0 0 1 {q[0]:.3f},{q[1]:.3f}")
    # 기록 스포크 (점선, 역 → DB)
    for k, (cx, cy) in enumerate(centers):
        if not stations[k][2]:
            continue
        L = math.hypot(cx - C[0], cy - C[1])
        u = ((cx - C[0]) / L, (cy - C[1]) / L)

        def bd(hx, hy):
            ds = [v for v in ((hx / abs(u[0])) if abs(u[0]) > 1e-9 else None,
                              (hy / abs(u[1])) if abs(u[1]) > 1e-9 else None) if v is not None]
            return min(ds)
        ds, dh = bd(sw / 2, sh / 2), bd(hw / 2, hh / 2)
        st = (cx - ds * u[0], cy - ds * u[1])
        en = (C[0] + (dh + 6) * u[0], C[1] + (dh + 6) * u[1])
        o += arrow(s, f"M{st[0]:.3f},{st[1]:.3f} L{en[0]:.3f},{en[1]:.3f}", "soft", dashed=True)
    # 역
    for k, (cx, cy) in enumerate(centers):
        name, sub, _ = stations[k]
        o += node(cx - sw // 2, cy - sh // 2, sw, sh, "focal" if k == focal else "card", name, sub)
    # 허브
    hx, hy = C[0] - hw // 2, C[1] - hh // 2
    o += f'<rect class="hub" x="{hx}" y="{hy}" width="{hw}" height="{hh}" rx="8"/>\n'
    o += (f'<text class="mono t-hub" x="{C[0]}" y="{C[1] - 2}" font-size="16" font-weight="600" '
          f'text-anchor="middle">mentions.db</text>\n')
    o += (f'<text class="sans t-hub" x="{C[0]}" y="{C[1] + 20}" font-size="12" text-anchor="middle" '
          f'opacity="0.8">판단·채점 원장</text>\n')
    o += legend(s, 648, 56, 632, [("solid", "매일 순서"), ("dashed", "DB에 기록"), ("focal", "통과해야 다음으로")])
    return s, o + "</svg>\n"


# ───────────── ③ 규칙 검증 단계 ─────────────
def rule_gate():
    s, W, H = "rule-gate", 920, 392
    w, h = 112, 64
    o = head(s, W, H, "규칙 검증 단계",
             "스크리닝 규칙이 초안으로 등록된 뒤 게이트를 통과하면 섀도 운영과 운영 투입으로, 기준에 못 미치면 "
             "탈락으로 가는 상태 전이. 게이트 판정은 지금 동작하고 이후 단계는 운영 연결 예정이다.")
    o += eyebrow(40, 36, "SCREENING RULE LIFECYCLE")
    o += ('<text class="sans t-muted" x="144" y="76" font-size="12">'
          '값 하나만 바꿔도 새 규칙 <tspan class="mono">(spec_hash)</tspan></text>\n')
    o += ('<text class="sans t-muted" x="352" y="212" font-size="12">통과선은 분기마다 미리 고정</text>\n')
    # 화살표
    o += arrow(s, "M46,128 H142")
    o += arrow(s, "M256,128 H350", "acc")
    o += arrow(s, "M464,128 H558", "soft", dashed=True)
    o += arrow(s, "M672,128 H766", "soft", dashed=True)
    o += arrow(s, "M200,160 V254")
    o += arrow(s, "M824,160 V254", "soft", dashed=True)
    o += arrow(s, "M616,160 V280 Q616,288 624,288 H766", "soft", dashed=True)
    # 라벨
    o += label(62, 104, 64, "규칙 등록", "t-muted")
    o += label(266, 104, 76, "게이트 통과", "t-acc", weight="600")
    o += label(480, 104, 64, "실전 관찰")
    o += label(688, 104, 64, "운영 교체")
    o += label(208, 200, 64, "기준 미달", "t-muted", anchor="start")
    o += label(832, 200, 32, "중단", anchor="start")
    o += label(680, 264, 32, "중단")
    # 노드
    o += '<circle class="dot" cx="40" cy="128" r="6"/>\n'
    o += node(144, 96, w, h, "card", "초안", "draft", True)
    o += node(352, 96, w, h, "focal", "통과", "gated", True)
    o += node(560, 96, w, h, "card", "섀도 운영", "shadow", True)
    o += node(768, 96, w, h, "card", "운영 투입", "active", True)
    o += node(144, 256, w, h, "store", "탈락", "rejected", True)
    o += node(768, 256, w, h, "store", "은퇴", "retired", True)
    o += legend(s, 368, 40, 880, [("solid", "지금 동작"), ("dashed", "운영 연결 예정")])
    return s, o + "</svg>\n"


for fn in (daily_flow, feedback_loop, rule_gate):
    slug, svg = fn()
    (OUT / f"{slug}.svg").write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + svg, encoding="utf-8")
    print(f"{slug}.svg  {len(svg):,} bytes")
