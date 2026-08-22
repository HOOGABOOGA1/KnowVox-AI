from app.rag.document_loader import load_pdf

print("Starting PDF test...")

pdf_path = "data/test.pdf"

print(f"Looking for PDF: {pdf_path}")

try:
    text = load_pdf(pdf_path)

    print("PDF loaded successfully!")
    print(f"Characters extracted: {len(text)}")

    print("\n--- EXTRACTED TEXT ---")
    print(text[:2000])

except Exception as e:
    print("\nERROR:")
    print(type(e).__name__, e)