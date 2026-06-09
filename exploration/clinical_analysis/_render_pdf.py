import sys
import fitz  # PyMuPDF

pdf = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\arman\Documents\tiagobot-bci-conference\main.pdf"
out_dir = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\arman\Documents\tiagobot-bci-conference\_pdfpreview"
import os
os.makedirs(out_dir, exist_ok=True)
doc = fitz.open(pdf)
print(f"{pdf}: {doc.page_count} pages")
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=110)
    p = os.path.join(out_dir, f"page{i+1:02d}.png")
    pix.save(p)
    print("wrote", p)
