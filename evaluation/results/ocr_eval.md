OCR accuracy on 6 real pages from Data_Act_2023_2854.pdf, GDPR_Regulation_2016_679.pdf, ePrivacy_Directive_2002_58.pdf (seed 42). Ground truth: the digital text of the same page.

| Condition | Pages | CER | WER | Word recall | Mean OCR confidence |
|---|---|---|---|---|---|
| clean scan, 200 dpi | 6 | 0.006 | 0.019 | 0.985 | 94.2 % |
| noisy scan | 6 | 0.018 | 0.037 | 0.981 | 93.3 % |
| skewed and blurred | 6 | 0.015 | 0.029 | 0.984 | 92.9 % |
| low contrast, JPEG 35, 150 dpi | 6 | 0.007 | 0.028 | 0.978 | 92.2 % |

Fallback with OCR switched off:
* a PDF that consists only of scans: rejected: No text could be extracted from this PDF (image-only PDF and OCR unavailable or empty). The document was not indexed.
* a mixed PDF: scanned page flagged = True, digital page still indexed = True; document warning: "1 page(s) have no text layer and need OCR."
