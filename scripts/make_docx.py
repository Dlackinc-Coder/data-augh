#!/usr/bin/env python
"""make_docx.py — genera COMPARATIVA_VISUAL_16OCT.docx (python-docx).

Uso (desde la raíz del proyecto):  python scripts/make_docx.py
Salida: docs/COMPARATIVA_VISUAL_16OCT.docx
"""
import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

IMGDIR = "docs/img"
OUT = "docs/COMPARATIVA_VISUAL_16OCT.docx"
BLUE = RGBColor(0x2E, 0x6F, 0xDB)
GREEN = RGBColor(0x2E, 0x9E, 0x5B)
GRAY = RGBColor(0x5F, 0x63, 0x68)


def set_style(doc):
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    rpr = st.element.get_or_add_rPr()
    rf = rpr.get_or_add_rFonts()
    rf.set(qn("w:eastAsia"), "Calibri")


def table(doc, data, widths=None):
    t = doc.add_table(rows=len(data), cols=len(data[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            r = p.add_run(str(val))
            r.font.size = Pt(10)
            if i == 0:
                r.font.bold = True
    if widths:
        for j, w in enumerate(widths):
            for i in range(len(data)):
                t.cell(i, j).width = Inches(w)
    return t


def pic(doc, fname, width_in=6.2):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(os.path.join(IMGDIR, fname), width=Inches(width_in))
    return p


def note(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(9)
    r.font.italic = True
    r.font.color.rgb = GRAY
    return p


doc = Document()
set_style(doc)

tp = doc.add_paragraph()
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tp.add_run("Comparativa visual — MioClinic vs DUCNet")
r.font.size = Pt(20)
r.font.bold = True
r.font.color.rgb = BLUE

sp = doc.add_paragraph()
sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sp.add_run("16-oct-2026 · mismo dataset y split que el paper (ratón cardiaco, 171/35/37)")
r.font.size = Pt(11)
r.font.color.rgb = GRAY

rp = doc.add_paragraph()
rp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = rp.add_run("Resultado: ")
r.font.size = Pt(14)
r.font.bold = True
r.font.color.rgb = GRAY
r = rp.add_run("AVG3 81.21 vs DUCNet 80.68  (+0.53)")
r.font.size = Pt(14)
r.font.bold = True
r.font.color.rgb = GREEN

doc.add_heading("1. El problema → el fix", level=1)
doc.add_paragraph(
    "El protocolo antiguo medía 8-30 tiles aleatorios por imagen: números inflados (~12 puntos) "
    "y no comparables con el paper. Todo se re-midió con ventana deslizante sobre la imagen "
    "completa (kit honesto, igual para todos)."
)
table(doc, [
    ["Ejemplo (A-v2)", "Protocolo viejo (tiles)", "Protocolo honesto", "Brecha"],
    ["Dice AVG", "89.7 %", "77.6 %", "-12.1 pts"],
], widths=[1.8, 1.8, 1.6, 1.2])

doc.add_heading("2. Resultado final vs DUCNet (test)", level=1)
table(doc, [
    ["Clase", "MioClinic (final)", "DUCNet", "Δ"],
    ["LV", "84.83", "84.22", "+0.61"],
    ["AAR", "75.70", "77.69", "-1.99"],
    ["IS", "83.11", "80.13", "+3.00"],
    ["AVG3", "81.21", "80.68", "+0.53"],
], widths=[1.2, 1.9, 1.5, 1.2])
pic(doc, "final_vs_ducnet.png")
note(doc, "HD/HD95 no comparables directamente (granularidad del paper no especificada). "
          "Detalle y análisis en docs/HALLAZGOS_16OCT.md §5.")

doc.add_heading("3. ¿Funciona de verdad? — 4 casos reales del test", level=1)
doc.add_paragraph(
    "Selección por percentil de AVG3 sobre las 37 imágenes de test (sin cherry-picking): "
    "percentil 85, mediana, percentil 15 y —por transparencia— el peor caso. Recorte "
    "alrededor del corazón; la predicción incluye la limpieza final (el mismo protocolo "
    "del 81.21). Colores: LV verde · AAR azul · IS rojo. Los Dice de cada pie están "
    "recalculados con el kit honesto sobre la predicción limpia (sanity-check: media de "
    "los 37 = 81.21 exacto)."
)
pic(doc, "caso_01_p85.png", 6.4)
pic(doc, "caso_02_mediana.png", 6.4)
pic(doc, "caso_03_p15.png", 6.4)
pic(doc, "caso_04_peor.png", 6.4)
note(doc, "El peor caso (id 236) muestra un fallo real: la IA sobre-predice LV (verde) donde "
          "el GT tiene AAR (azul) — LV 2.7. Incluso ahí, el IS (la clase donde ganamos al "
          "SOTA, +3.00) se detecta bien (81.8).")

doc.add_heading("4. El camino al cierre", level=1)
pic(doc, "camino.png")
note(doc, "Ensemble U-Net + Swin (α=0.5) + TTA flips×4 + limpieza de islas (<4000 px). "
          "Elegido en validación con regla pre-registrada; aplicado una sola vez en test (val 83.40).")

doc.add_heading("5. Panorama completo (test, protocolo honesto)", level=1)
pic(doc, "panorama.png", 6.4)

doc.add_heading("6. Cambios clave del proyecto", level=1)
table(doc, [
    ["#", "Cambio", "Para qué"],
    ["1", "Kit de evaluación honesta full-image", "Medir lo mismo que el paper (sin sesgo de tiles)"],
    ["2", "Fix del freeze del encoder (era no-op)", "Transfer learning real en las U-Net"],
    ["3", "Fix del scheduler", "LR decay correcto en todos los runs"],
    ["4", "bf16 verificado == fp32", "2× más rápido, sin pérdida de precisión"],
    ["5", "Limpieza por componentes (r=2, 4000 px)", "Elimina islas: HD95 -9…-211 px en val"],
    ["6", "TTA flips×4", "+0.26…+0.78 pts en val"],
    ["7", "Ensembles CNN+Transformer", "4/4 superan a DUCNet por sí solos"],
    ["8", "Selección en val + test one-shot", "Defensa metodológica (sin tuning en test)"],
], widths=[0.5, 3.0, 3.0])

doc.add_paragraph()
note(doc, "Fase 3 preparada (Tversky, selección de checkpoint por val): configs listas.")

fp = doc.add_paragraph()
r = fp.add_run("Rutas:  docs/HALLAZGOS_16OCT.md  ·  docs/COMPARATIVA_VISUAL_16OCT.md  ·  "
               "runs/eval_honest/SUMMARY.md")
r.font.size = Pt(9)
r.font.color.rgb = GRAY

doc.save(OUT)
print("ok:", OUT)
