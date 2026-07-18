.PHONY: dev test migrate seed logs down

dev:
	docker compose up --build

test:
	docker compose run api pytest tests/ -v

migrate:
	docker compose run api alembic upgrade head

seed:
	docker compose run api python -c "from src.database import seed_demo_users; import asyncio; asyncio.run(seed_demo_users())"

logs:
	docker compose logs -f api

down:
	docker compose down -v
