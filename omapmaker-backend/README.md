# OMapMaker Backend

FastAPI backend pro generování orientačních map z LiDAR dat ČÚZK.

## Struktura projektu

```
omapmaker-backend/
├── app/
│   ├── main.py              ← FastAPI app + CORS
│   ├── core/
│   │   ├── downloader.py    ← Stahování ČÚZK ATOM dlaždic
│   │   ├── processor.py     ← DTM/DSM interpolace, vegetace, skály
│   │   ├── renderer.py      ← Vrstevnice, matplotlib → PNG
│   │   ├── symbols.py       ← Načítání symbols.xml, kreslení ISOM
│   │   ├── vector_layers.py ← OSM/ZABAGED/ISOM → ISOM symboly
│   │   ├── pipeline.py      ← Orchestrace celého jobu
│   │   └── exporter.py      ← Export do GPKG pro OOM
│   └── routes/
│       ├── jobs.py          ← POST/GET /api/jobs
│       └── download.py      ← POST /api/download/cuzk
├── requirements.txt
├── symbols10.xml            ← ISOM symboly 1:10 000 (zkopíruj sem!)
├── symbols15.xml            ← ISOM symboly 1:15 000 (zkopíruj sem!)
└── README.md
```

## Instalace

### 1. Python prostředí

```bash
# Python 3.11+ doporučeno
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 2. Symbols.xml

Zkopíruj soubory `symbols10.xml` a `symbols15.xml` z původní aplikace
do kořenového adresáře backendu (vedle `requirements.txt`).

### 3. Spuštění

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API bude dostupné na `http://localhost:8000`.

Swagger dokumentace: `http://localhost:8000/docs`

## API endpointy

### POST /api/jobs
Spustí generování mapy.

**Form data:**
- `dtm` — DTM soubor (.las/.laz)
- `dsm` — DSM soubor (.las/.laz/.tif)
- `zabaged` — ZABAGED® soubory (.shp), volitelné, více souborů
- `isom` — Vlastní ISOM vrstvy (.shp), volitelné, více souborů
- `params` — JSON string s parametry:

```json
{
  "crs": "EPSG:5514",
  "scale": 10000,
  "paper_format": "A4 (Landscape)",
  "sigma": 6.5,
  "slope_threshold": 45.0,
  "north_rotation": 5.0,
  "bins": [1, 2, 6, 12],
  "layers": {
    "contours": true,
    "rocks": true,
    "water": true,
    "vegetation": true,
    "roads": true,
    "buildings": true,
    "man_made": true,
    "magnetic_lines": false
  }
}
```

**Odpověď:** `{ "job_id": "abc12345" }`

---

### GET /api/jobs/{job_id}
Vrátí stav jobu.

```json
{
  "job_id": "abc12345",
  "status": "running",     // queued | running | done | error
  "progress": 62,          // 0-100
  "step": "Generuji vrstevnice...",
  "error": null
}
```

---

### GET /api/jobs/{job_id}/png
Stáhne vygenerovanou PNG mapu (1000 DPI).

### GET /api/jobs/{job_id}/gpkg
Stáhne GPKG export pro OpenOrienteering Mapper.

---

### POST /api/download/cuzk
Stáhne LiDAR dlaždice z ČÚZK ATOM pro zadanou oblast.

```json
{
  "bbox": { "min_lat": 50.1, "min_lon": 15.8, "max_lat": 50.3, "max_lon": 16.1 },
  "dsm_type": "DMPOK",
  "out_dir": "./cuzk_data"
}
```

**Odpověď:** `{ "dmr_path": "...", "dmp_path": "..." }`

## Nasazení

### Lokálně (vývoj)
```bash
uvicorn app.main:app --reload
```

### Produkce (Railway / Render / VPS)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Na Railway: nastav environment variable `OMAPMAKER_JOBS_DIR=/tmp/jobs`.

**Poznámka:** Pro produkci s více uživateli doporučujeme nahradit
threading.Thread za Celery + Redis nebo ProcessPoolExecutor.
