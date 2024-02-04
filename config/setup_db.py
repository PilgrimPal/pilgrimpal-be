import asyncio
from databases import Database
from settings import get_settings


async def setup_db():

    settings = get_settings()
    db = Database(settings.DATABASE_URL)
    await db.connect()

    with open("config/ddl.sql") as ddl:
        query = ddl.read()
        for q in query.split(";"):
            print(q)
            await db.execute(query=q + ";")

    await db.disconnect()

    print("Done")


if __name__ == "__main__":
    asyncio.run(setup_db())
