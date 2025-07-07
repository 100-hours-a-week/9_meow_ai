from fastapi import APIRouter
from ai_server.router.posts import router as posts_router
from ai_server.router.comments import router as comments_router
from ai_server.router.images import router as images_router

api_router = APIRouter()

api_router.include_router(
    posts_router, 
    prefix="/generate/post", 
    tags=["Posts"]
)

api_router.include_router(
    comments_router, 
    prefix="/generate/comment", 
    tags=["Comments"]
)

api_router.include_router(
    images_router,
    prefix="/images",
    tags=["Images"]
)