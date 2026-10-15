from fastapi import FastAPI
from APP.routers.answer import router

app = FastAPI()
app.include_router(router)