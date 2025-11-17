from fastapi import APIRouter, Body, Depends, HTTPException
from db.engine import get_db
from typing import Annotated
from sqlalchemy.ext.asyncio import AsyncSession
from services.services import (
    UserOperation,
    TextOperation,
    ArticleOperation,
    TranslationOperation,
    FileOperation,
)
from schema.jwt import JWTPayload
from utils.jwt import JWTHandler
from fastapi import Query
from schema._input import (
    UserInput,
    UpdateUserProfileInput,
    UserInputLogin,
    TextCreateRequest,
)
from uuid import UUID
import logging

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.post("/register")
async def register_user(
    db_session: Annotated[AsyncSession, Depends(get_db)], data: UserInput = Body(...)
):
    user_operation = UserOperation(db_session)
    user = await user_operation.register_user(
        username=data.username,
        phone_number=data.phone_number,
        degree=data.degree,
        password=data.password,
        confirm_password=data.confirm_password,
    )
    return user


@router.post("/login")
async def login_user(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    data: UserInputLogin = Body(...),
):
    user_operation = UserOperation(db_session)
    token = await user_operation.login_user(data.username, data.password)
    return (token,)


@router.get("/{username}/")
async def get_user_profile(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    username: str,
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    user_operation = UserOperation(db_session)
    user_profile = await user_operation.get_user_by_username(username)
    return user_profile


@router.put("/")
async def user_update_profile(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    data: UpdateUserProfileInput = Body(...),
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    user_operation = UserOperation(db_session)
    user = await user_operation.update_user(
        username=token_data.username,
        family=data.family,
        age=data.age,
        phone_number=data.phone_number,
        degree=data.degree,
        email=data.email,
    )
    return user


@router.delete("/")
async def delete_user_account(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    user_operation = UserOperation(db_session)
    await user_operation.delete_user(token_data.username)
    return {"detail": "User account deleted successfully."}


@router.post("/texts/")
async def create_text(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    user_id: UUID = Query(...),
    request: TextCreateRequest = Body(...),
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    text_operation = TextOperation(db_session)
    return await text_operation.create_text(user_id, request.text_content)


@router.delete("/texts/{text_id}")
async def delete_text(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    text_id: UUID,
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    text_operation = TextOperation(db_session)
    await text_operation.delete_text(text_id)
    return {"detail": "Text deleted successfully."}


@router.put("/texts/{text_id}")
async def update_text(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    text_id: UUID,
    new_text: str = Body(...),
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    text_operation = TextOperation(db_session)
    return await text_operation.update_text(text_id, new_text)


@router.get("/texts/{user_id}")
async def list_user_texts(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    user_id: UUID,
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    text_operation = TextOperation(db_session)
    return await text_operation.list_user_texts(user_id)


@router.delete("/articles/{article_id}")
async def delete_article(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    article_id: UUID,
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    article_operation = ArticleOperation(db_session)
    await article_operation.delete_article(article_id)
    return {"detail": "Article deleted successfully."}


@router.get("/articles/{user_id}")
async def list_user_generated_articles(
    user_id: UUID,
    db_session: Annotated[AsyncSession, Depends(get_db)],
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    article_operation = ArticleOperation(db_session)
    generated_articles = await article_operation.list_user_generated_articles(user_id)
    if not generated_articles:
        raise HTTPException(status_code=404, detail="No generated articles found.")
    return generated_articles


@router.post("/articles/process")
async def process_article(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    user_id: UUID,
    text_id: UUID,
    header: str = Body(...),
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    article_operation = ArticleOperation(db_session)
    generated_article = await article_operation.generate_article(
        user_id, text_id, header
    )
    return {"generated_article": generated_article}


@router.post("/articles/{article_id}/refine")
async def refine_article(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    article_id: UUID,
    user_id: UUID,
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    article_operation = ArticleOperation(db_session)
    refined_article = await article_operation.refine_article(article_id, user_id)
    return {"refined_article": refined_article}


@router.post("/articles/{article_id}/evaluate")
async def evaluate_article(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    article_id: UUID,
    user_id: UUID,
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    article_operation = ArticleOperation(db_session)
    evaluation = await article_operation.evaluate_article(article_id, user_id)
    return {"evaluation": evaluation}


@router.post("/articles/{article_id}/generate_pdf")
async def generate_pdf_article(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    article_id: UUID,
    user_id: UUID,
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    file_operation = FileOperation(db_session)
    pdf_data = await file_operation.generate_pdf(article_id, user_id)
    return {"download_url": pdf_data["download_url"]}


@router.post("/articles/{article_id}/generate_word")
async def generate_word_article(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    article_id: UUID,
    user_id: UUID,
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    file_operation = FileOperation(db_session)
    word_data = await file_operation.generate_word(article_id, user_id)
    return {"download_url": word_data["download_url"]}


@router.post("/articles/{article_id}/generate_latex")
async def generate_latex_article(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    article_id: UUID,
    user_id: UUID,
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    file_operation = FileOperation(db_session)
    latex_data = await file_operation.generate_latex(article_id, user_id)
    return {"download_url": latex_data["download_url"]}


@router.get("/files/{file_id}/")
async def get_generated_pdf(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    file_id: UUID,
    user_id: UUID,
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    file_operation = FileOperation(db_session)
    file_data = await file_operation.get_file(file_id, user_id)
    return file_data


@router.post("/reset-password")
async def reset_password(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
    new_password: str = Body(..., embed=True),
    confirm_password: str = Body(..., embed=True),
):
    if new_password != confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")

    user_operation = UserOperation(db_session)
    result = await user_operation.reset_password_by_username(
        token_data.username, new_password
    )
    return {"detail": "Password updated successfully."}


@router.post("/forget-password/request")
async def request_recovery_code(
    username: str = Body(..., embed=True),
    db_session: AsyncSession = Depends(get_db),
):
    forget_service = UserOperation(db_session)
    recovery_code = await forget_service.generate_recovery_code(username)
    return {"recovery_code": recovery_code}


@router.post("/forget-password/confirm")
async def confirm_recovery_code_and_reset_password(
    username: str = Body(..., embed=True),
    recovery_code: str = Body(..., embed=True),
    new_password: str = Body(..., embed=True),
    db_session: AsyncSession = Depends(get_db),
):
    forget_service = UserOperation(db_session)
    user_operation = UserOperation(db_session)
    result = await forget_service.verify_recovery_code_and_reset_password(
        username, recovery_code, new_password, user_operation
    )
    return result


@router.post("/articles/{article_id}/translate")
async def translate_article(
    article_id: UUID,
    db_session: Annotated[AsyncSession, Depends(get_db)],
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):
    service = TranslationOperation(db_session)
    result = await service.translate_article(article_id)
    return result


@router.post("/feedback")
async def submit_feedback(
    db_session: Annotated[AsyncSession, Depends(get_db)],
    article_id: UUID = Body(...),
    feedback: str = Body(...),
    score: int = Body(...),
    token_data: JWTPayload = Depends(JWTHandler.verify_token),
):

    logger.info(
        f"Feedback for article {article_id}: {feedback}, Score: {score}, User: {token_data.username}"
    )
    return {"detail": "Feedback submitted successfully."}
