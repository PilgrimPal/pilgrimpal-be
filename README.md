# PilgrimPal Backend

## Development

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Start FastAPI process

```bash
uvicorn main:app --reload
```

3. Open local API docs [http://localhost:8000/docs](http://localhost:8000/docs)

## Docker

1. Run

```bash
docker compose up -d --build
```

2. .env for db's host can't be localhost so change it to `POSTGRES_HOST=pp-db`

## Setup DB

1. Run (To execute the required tables' DDL)
```bash
# if you're running this on local, don't forget to change the DB .env to host=localhost and port=6543
python config/setup_db.py
```