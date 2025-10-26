# 🧠Testing Smart Recruitment Service by NLP Pretrained Model

## ⚙️ Langkah Instalasi
###  Clone Repository
```bash
git clone https://github.com/wgalambarus/SmartRec/.git
cd folder-projek
```
### Setting Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
### Project Structure
```bash
smart-recruitment/
│
├── python.py                # Main script
├── requirements.txt         # List dependensi
├── job.txt                  # Deskripsi pekerjaan
├── cvs/                     # Folder berisi file CV kandidat
│   ├── cv1.pdf
│   ├── cv2.pdf
│   └── cv3.pdf
└── .gitignore
```
### How to Run
```bash
python python.py --cv_folder "cvs" --job_desc "job.txt" --out "ranking.csv"
```

NOTES : Jangan lupa add folder cvs yang berisi cv.pdf


