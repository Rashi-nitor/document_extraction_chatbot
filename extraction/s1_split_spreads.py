"""Step 1 - split two-page spreads into single printed pages.

Why: most catalogue pages are spreads (left + right page on one PDF page).
Layout models read a spread as one wide page and mix both halves, so we cut
each spread in two first. A plain crop is not enough - the hidden half would
still be in the text layer - so the other half is redacted before cropping.

Output: data/interim/catalogue_split.pdf (60 pages = printed pages 1-60).
"""
import pymupdf

from common import INPUT_PDF, SPLIT_PDF, iter_half_pages


def main():
    src = pymupdf.open(INPUT_PDF)
    out = pymupdf.open()
    for idx, printed, page, clip in iter_half_pages(src):
        out.insert_pdf(src, from_page=idx, to_page=idx)
        new = out[-1]
        full = new.rect
        if clip != full:  # spread half: wipe the other half, then crop
            other = (pymupdf.Rect(clip.x1, 0, full.x1, full.y1) if clip.x0 == 0
                     else pymupdf.Rect(0, 0, clip.x0, full.y1))
            new.add_redact_annot(other)
            new.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                                 graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
            new.set_cropbox(clip)
    SPLIT_PDF.parent.mkdir(parents=True, exist_ok=True)
    out.save(SPLIT_PDF, garbage=3, deflate=True)
    print(f"split {len(src)} PDF pages -> {len(out)} printed pages: {SPLIT_PDF}")


if __name__ == "__main__":
    main()
