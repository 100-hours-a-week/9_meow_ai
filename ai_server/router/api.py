from fastapi import APIRouter
from ai_server.router.api import posts
from ai_server.router.api import comments

api_router = APIRouter()

api_router.include_router(
    posts.router, 
    prefix="/generate/post", 
    tags=["Posts"]
)

api_router.include_router(
    comments.router, 
    prefix="/generate/comment", 
    tags=["Comments"]
) 