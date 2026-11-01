"""Build REPORT.pdf from results/ (metrics, experiment log, figures) and report_text.md (narrative).
report_text.md uses a tiny markup: '# ' h1, '## ' h2, '- ' bullet, '[[fig:path|caption|width_cm]]' figure,
'[[table:explog]]', '[[table:summary]]', '[[table:split]]' generated tables; blank line = paragraph break."""
import os, json, re
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from PIL import Image as PILImage
from common import ROOT

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=12.5, spaceAfter=4)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8, textColor=colors.HexColor("#444444"), alignment=1)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=6.6, leading=8, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")

def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", t)
    t = re.sub(r"(https?://[^\s)]+)", r"<link href='\1' color='blue'>\1</link>", t)
    return t

def fig(path, caption, width_cm):
    p = os.path.join(ROOT, path); w, h = PILImage.open(p).size
    W = float(width_cm) * cm; H = W * h / w
    maxH = 23 * cm
    if H > maxH: W, H = W * maxH / H, maxH
    return KeepTogether([Image(p, W, H), Paragraph(inline(caption), CAP), Spacer(1, 6)])

def table(rows, widths, header=True, font=6.6):
    data = [[Paragraph(inline(str(c)), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 2.5), ("RIGHTPADDING", (0, 0), (-1, -1), 2.5)]))
    return t

def explog_table():
    d = pd.read_csv(os.path.join(ROOT, "results/experiment_log.csv"))
    dp = os.path.join(ROOT, "results/experiment_decisions.json")
    decisions = json.load(open(dp)) if os.path.exists(dp) else {}
    rows = [["Trial", "What changed and why", "Params (overrides vs previous / base)", "IoU", "Dice", "Prec", "Rec", "BF", "Inst F1", "Count MAE", "s/img", "Decision"]]
    for _, r in d.iterrows():
        why = str(r["change_and_reason"]); dec = decisions.get(str(r["trial"]), "")
        rows.append([r["trial"], why, r["config_overrides"], f"{r['mean_iou']:.3f}", f"{r['mean_dice']:.3f}", f"{r['mean_precision']:.3f}",
                     f"{r['mean_recall']:.3f}", f"{r['mean_bf']:.3f}", f"{r['inst_f1']:.3f}", f"{r['count_mae']:.2f}", f"{r['sec_per_img']:.2f}", dec])
    return table(rows, [0.9, 4.4, 4.4, 0.85, 0.85, 0.85, 0.85, 0.85, 0.9, 0.9, 0.75, 2.6])

def summary_table():
    s = json.load(open(os.path.join(ROOT, "results/metrics_summary.json")))
    keys = [("mean_iou", "Mean IoU (per image)"), ("mean_dice", "Mean Dice"), ("mean_precision", "Mean pixel precision"), ("mean_recall", "Mean pixel recall"),
            ("mean_bf", "Mean boundary F (tol 9 px)"), ("pooled_iou", "Pooled IoU (all pixels)"), ("pooled_dice", "Pooled Dice"),
            ("inst_precision", "Instance precision (IoU>=0.5)"), ("inst_recall", "Instance recall (IoU>=0.5)"), ("inst_f1", "Instance F1"),
            ("count_mae", "Count MAE (eggs/image)"), ("count_exact_acc", "Exact count accuracy"), ("n_images", "Images")]
    rows = [["Metric", "Dev", "Eval (held-out)"]]
    for k, n in keys:
        f = (lambda v: str(v)) if k == "n_images" else (lambda v: f"{v:.3f}")
        rows.append([n, f(s["dev"][k]), f(s["eval"][k])])
    return table(rows, [6, 3, 3], font=8)

def split_table():
    sp = json.load(open(os.path.join(ROOT, "results/split.json")))
    sizes = sorted((len(g) for g in sp["groups"]), reverse=True)
    rows = [["Item", "Value"], ["Images", str(len(sp["dev"]) + len(sp["eval"]))], ["Scene groups", f"{len(sp['groups'])} (sizes {sizes[:10]} ...)"],
            ["Dev / Eval", f"{len(sp['dev'])} / {len(sp['eval'])}"], ["Grouping params", json.dumps(sp["params"])]]
    return table(rows, [4, 12], font=8)

BUL2 = ParagraphStyle("bul2", parent=BUL, leftIndent=26, bulletIndent=16)
LIST_RE = re.compile(r"^(\s*)(-|\d+\.)\s+(.*)$")

def list_or_para(b):
    """A block is a paragraph, a list, or a lead-in line followed by a list. '- ' items, '  - ' nested items and
    '1. ' numbered items each start a new entry; other lines continue the previous entry."""
    items = []  # (kind, level, bullet, text)
    for line in b.split("\n"):
        m = LIST_RE.match(line)
        if m:
            lvl = 1 if len(m.group(1)) >= 2 else 0
            items.append(("li", lvl, "•" if m.group(2) == "-" else m.group(2), m.group(3)))
        elif items:
            k, l, bu, t = items[-1]; items[-1] = (k, l, bu, t + " " + line.strip())
        else:
            items.append(("p", 0, None, line.strip()))
    out = []
    for k, l, bu, t in items:
        if k == "p": out.append(Paragraph(inline(t), BODY))
        else: out.append(Paragraph(inline(t), BUL2 if l else BUL, bulletText=bu))
    return out

def build():
    txt = open(os.path.join(ROOT, "report_text.md")).read()
    story = []
    for block in re.split(r"\n\s*\n", txt.strip()):
        b = block.strip()
        if b == "[[pagebreak]]": story.append(PageBreak()); continue
        if b.startswith("[[fig:"):
            path, cap, w = b[6:-2].split("|"); story.append(fig(path, cap, w)); continue
        if b == "[[table:explog]]": story.append(explog_table()); story.append(Spacer(1, 6)); continue
        if b == "[[table:summary]]": story.append(summary_table()); story.append(Spacer(1, 6)); continue
        if b == "[[table:split]]": story.append(split_table()); story.append(Spacer(1, 6)); continue
        if b.startswith("# "): story.append(Paragraph(inline(b[2:]), H1)); continue
        if b.startswith("## "): story.append(Paragraph(inline(b[3:]), H2)); continue
        story.extend(list_or_para(b))
    out = os.path.join(ROOT, "REPORT.pdf")
    def footer(c, d):
        c.setFont("Helvetica", 7); c.setFillColor(colors.grey)
        c.drawString(1.5 * cm, 1 * cm, "Nosh Robotics CV Intern task - Kshitij Kumar Jha"); c.drawRightString(19.5 * cm, 1 * cm, f"page {d.page}")
    SimpleDocTemplate(out, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.6 * cm,
                      title="Egg segmentation and tracking - Nosh Robotics CV task", author="Kshitij Kumar Jha").build(story, onFirstPage=footer, onLaterPages=footer)
    print(out)

if __name__ == "__main__":
    build()
