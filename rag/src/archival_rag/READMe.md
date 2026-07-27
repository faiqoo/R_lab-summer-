1. Install the one dependency:
pip install pdfplumber
2. Put both scripts in the same folder, e.g.:
project/
├── pdf_to_chunks_v3.py
├── batch_chunk_pdfs.py
└── pdfs/              <- put all your regulation PDFs here
3. Run the batch script, pointing at your PDF folder:
python batch_chunk_pdfs.py ./pdfs ./output

First argument = folder containing your PDFs
Second argument = folder where results get written (created automatically if it doesn't exist)
4. output:
all_chunks.json
processing_report.json