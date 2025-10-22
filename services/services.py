from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy.future import select
from deep_translator import GoogleTranslator
from uuid import UUID
from datetime import datetime
from fastapi import HTTPException
from db.models import UserModel, TextModel, ArticleModel, AIModel, FileModel
from utils.secrets import password_manager
from schema._output import RegisterOutput
from utils.jwt import JWTHandler
from utils.storage import save_file
from uuid import uuid4
from docx import Document
from fpdf import FPDF
from io import BytesIO
from utils.keys import *
import numpy as np
import requests
import faiss
import bcrypt


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed_password.decode("utf-8")


def verify_password(stored_password: str, provided_password: str) -> bool:
    return bcrypt.checkpw(
        provided_password.encode("utf-8"), stored_password.encode("utf-8")
    )


class UserOperation:
    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def register_user(
        self,
        username: str,
        phone_number: str,
        degree: str,
        password: str,
        confirm_password: str,
    ) -> RegisterOutput:
        if password != confirm_password:
            raise HTTPException(
                status_code=400,
                detail="Passwords do not match.",
            )

        async with self.db_session as session:
            try:
                print(
                    f"Attempting to register user with username: {username} and phone_number: {phone_number}"
                )

                existing_user = await session.execute(
                    select(UserModel).where(
                        (UserModel.username == username)
                        | (UserModel.phone_number == phone_number)
                    )
                )
                if existing_user.scalars().first():
                    raise HTTPException(
                        status_code=400,
                        detail="User with this username or phone number already exists.",
                    )

                new_user = UserModel(
                    username=username,
                    phone_number=phone_number,
                    password=password_manager.hash(password),
                    degree=degree,
                    recovery_code=None,
                    email=None,
                    age=None,
                    family=None,
                    profile_image=None,
                )
                session.add(new_user)
                await session.commit()
                return RegisterOutput(username=new_user.username, id=new_user.id)
            except IntegrityError as e:
                await session.rollback()
                print(f"IntegrityError: {e.orig}")
                raise HTTPException(
                    status_code=400,
                    detail="User with this username or phone number already exists.",
                )

    async def login_user(self, username: str, password: str) -> dict:
        async with self.db_session as session:
            user = await session.execute(
                select(UserModel).where(UserModel.username == username)
            )
            user = user.scalars().first()

            if user is None or not password_manager.verify(password, user.password):
                raise HTTPException(
                    status_code=400, detail="Invalid username or password."
                )

            token = JWTHandler.generate(username)
            return {"token": token, "user_id": user.id}

    async def get_user_by_username(self, username: str):

        stmt = select(UserModel).where(UserModel.username == username)
        result = await self.db_session.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found.")

        return user

    async def get_user_profile(self, user_identifier: str) -> dict:
        async with self.db_session as session:
            query = select(UserModel).where(
                (UserModel.id == user_identifier)
                | (UserModel.username == user_identifier)
            )
            user = await session.execute(query)
            user = user.scalars().first()

            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            return {
                "username": user.username,
                "name": user.name,
                "family": user.family,
                "profile_image": user.profile_image,
            }

    async def delete_user(self, username: str) -> None:
        async with self.db_session as session:
            user = await session.execute(
                select(UserModel).where(UserModel.username == username)
            )
            user = user.scalars().first()

            if user is None:
                raise HTTPException(status_code=404, detail="User not found.")

            await session.delete(user)
            await session.commit()

    async def update_user(
        self,
        username: str,
        family: str = None,
        age: int = None,
        phone_number: str = None,
        degree: str = None,
        email: str = None,
    ) -> UserModel:
        async with self.db_session as session:
            user = await session.execute(
                select(UserModel).where(UserModel.username == username)
            )
            user = user.scalars().first()

            if user is None:
                raise HTTPException(status_code=404, detail="User not found.")

            if family is not None:
                user.family = family
            if age is not None:
                user.age = age
            if phone_number is not None:
                user.phone_number = phone_number
            if degree is not None:
                user.degree = degree
            if email is not None:
                user.email = email

            await session.commit()
            return user

    async def reset_password_by_username(self, username: str, new_password: str):
        user = await self.get_user_by_username(username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")

        hashed_password = self.hash_password(new_password)
        user.password = hashed_password
        await self.db_session.commit()
        return {"message": "Password updated successfully."}

    def hash_password(self, password: str) -> str:
        from passlib.context import CryptContext

        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.hash(password)

    async def generate_recovery_code(self, username: str) -> str:
        stmt = select(UserModel).where(UserModel.username == username)
        result = await self.db_session.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found.")

        recovery_code = str(uuid4()).split("-")[0]
        user.recovery_code = recovery_code
        await self.db_session.commit()

        return recovery_code

    async def verify_recovery_code_and_reset_password(
        self,
        username: str,
        recovery_code: str,
        new_password: str,
        user_operation_service,
    ):
        stmt = select(UserModel).where(UserModel.username == username)
        result = await self.db_session.execute(stmt)
        user = result.scalar_one_or_none()

        if not user or user.recovery_code != recovery_code:
            raise HTTPException(status_code=400, detail="Invalid recovery code.")

        await user_operation_service.reset_password_by_username(username, new_password)
        user.recovery_code = None
        await self.db_session.commit()

        return {"detail": "Password reset successfully."}


class TextOperation:
    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def create_text(
        self, user_id: UUID, text_content: str, creation_date: datetime = None
    ) -> dict:
        async with self.db_session as session:
            user = await session.execute(
                select(UserModel).where(UserModel.id == user_id)
            )
            user = user.scalars().first()
            if user is None:
                raise HTTPException(status_code=404, detail="User not found.")

            if creation_date is None:
                creation_date = datetime.utcnow()

            new_text = TextModel(
                user_id=user.id, text=text_content, creation_date=creation_date
            )
            session.add(new_text)
            await session.commit()
            return {
                "success": "Text created successfully.",
                "text_id": str(new_text.id),
            }

    async def delete_text(self, text_id: UUID) -> None:
        async with self.db_session as session:
            text = await session.execute(
                select(TextModel).where(TextModel.id == text_id)
            )
            text = text.scalars().first()
            if text is None:
                raise HTTPException(status_code=404, detail="Text not found.")

            await session.delete(text)
            await session.commit()

    async def update_text(self, text_id: UUID, new_text: str) -> dict:
        async with self.db_session as session:
            text = await session.execute(
                select(TextModel).where(TextModel.id == text_id)
            )
            text = text.scalars().first()
            if text is None:
                raise HTTPException(status_code=404, detail="Text not found.")

            text.text = new_text
            await session.commit()
            return {"success": "Text updated successfully."}

    async def list_user_texts(self, user_id: UUID) -> list:
        async with self.db_session as session:
            result = await session.execute(
                select(TextModel).where(TextModel.user_id == user_id)
            )
            texts = result.scalars().all()

            if not texts:
                raise HTTPException(
                    status_code=404, detail="No texts found for the user."
                )

            result_list = []
            for text in texts:
                result_list.append(
                    {
                        "id": str(text.id),
                        "user_id": str(text.user_id),
                        "content": text.text,
                        "creation_date": text.creation_date,
                    }
                )
            return result_list


class ArticleRetriever:
    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session
        self.ai_api_key = AI_API_KEY
        self.index = None
        self.id_map = []

    async def build_faiss_index(self):
        async with self.db_session as session:
            articles = await session.execute(select(ArticleModel))
            articles = articles.scalars().all()

        article_vectors = [self.embed_article(article.content) for article in articles]
        if not article_vectors:
            raise Exception("No articles to build index")

        vectors = np.array(article_vectors).astype(np.float32)
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.id_map = [article.id for article in articles]

    def embed_article(self, text: str) -> np.ndarray:
        url = "https://api.cohere.ai/v1/embed"
        headers = {
            "Authorization": f"Bearer {self.ai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"texts": [text]}
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            embeddings = response.json().get("embeddings", [])
            return np.array(embeddings[0], dtype=np.float32)
        else:
            raise HTTPException(status_code=500, detail="Embedding failed")

    async def add_to_faiss_index(self, article_vector: np.ndarray, article_id: UUID):
        if self.index is None:
            self.index = faiss.IndexFlatL2(article_vector.shape[0])

        vector = np.array([article_vector]).astype(np.float32)
        faiss.normalize_L2(vector)
        self.index.add(vector)
        self.id_map.append(article_id)

    def retrieve_relevant_articles(self, query_vector: np.ndarray, top_k=5) -> list:
        if self.index is None:
            raise Exception("Index has not been built yet.")

        query_vector = np.array([query_vector]).astype(np.float32)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)

        retrieved_ids = []
        for idx in indices[0]:
            if idx < len(self.id_map):
                retrieved_ids.append(self.id_map[idx])

        return retrieved_ids


class ArticleGenerator:
    def __init__(self) -> None:
        self.api_key = AI_API_KEY

    def generate_article(
        self,
        user_text: str,
        relevant_articles: list,
    ) -> str:
        articles_content = "\n".join([article.content for article in relevant_articles])

        full_prompt = (
            "Using the following user-provided text and research articles, write an academic-style article strictly based on the given title.\n"
            "Avoid generic introductions or unrelated content. Stay focused on the core subject.\n\n"
            f"User text:\n{user_text}\n\n"
            f"Research articles:\n{articles_content}\n\n"
            "Article:"
        )

        url = "https://api.cohere.ai/v1/generate"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "prompt": full_prompt,
            "max_tokens": 6000,
            "temperature": 0.7,
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json().get("generations", [{}])[0].get("text", "")
        else:
            raise Exception("Failed to generate article")


class ArticleOperation:
    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session
        self.retriever = ArticleRetriever(db_session)
        self.generator = ArticleGenerator()

    async def extract_keywords(self, text_id: UUID) -> list:
        async with self.db_session as session:
            text = await session.execute(
                select(TextModel).where(TextModel.id == text_id)
            )
            text = text.scalars().first()
            if not text:
                raise HTTPException(status_code=404, detail="Text not found")

        url = "https://api.cohere.ai/v1/generate"
        headers = {
            "Authorization": f"Bearer {self.retriever.ai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": f"Extract clear, topic-focused keywords from this academic text:\n{text.text}",
            "max_tokens": 80,
        }
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            return (
                response.json().get("generations", [{}])[0].get("text", "").split(", ")
            )
        else:
            raise HTTPException(status_code=500, detail="Keyword extraction failed")

    async def search_articles_with_crossref(self, keywords: list) -> list:
        search_results = []
        joined_keywords = " ".join(keywords)
        url = f"https://api.crossref.org/works?query={joined_keywords}&rows=40"

        response = requests.get(url)
        if response.status_code == 200:
            results = response.json().get("message", {}).get("items", [])
            for item in results:
                abstract = item.get("abstract")
                title = item.get("title", [None])[0]
                if title and abstract:
                    search_results.append(
                        {
                            "title": title,
                            "link": item.get("URL", "No link"),
                            "snippet": abstract,
                        }
                    )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"CrossRef API failed: {response.status_code} - {response.text}",
            )

        return search_results

    async def generate_article(self, user_id: UUID, text_id: UUID, header: str) -> dict:
        async with self.db_session as session:
            user = await session.execute(
                select(UserModel).where(UserModel.id == user_id)
            )
            user = user.scalars().first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            text = await session.execute(
                select(TextModel).where(TextModel.id == text_id)
            )
            text = text.scalars().first()
            if not text:
                raise HTTPException(status_code=404, detail="Text not found")

        keywords = await self.extract_keywords(text_id)
        articles_from_crossref = await self.search_articles_with_crossref(keywords)

        valid_articles = []
        async with self.db_session as session:
            for article in articles_from_crossref:
                new_article = ArticleModel(
                    title=article["title"],
                    header=header,
                    content=article["snippet"],
                    link=article["link"],
                    snippet=article["snippet"],
                    user_id=user_id,
                    text_id=text_id,
                    created_at=datetime.utcnow(),
                )
                session.add(new_article)
                valid_articles.append(new_article)
            await session.commit()

        for article in valid_articles:
            vector = self.retriever.embed_article(article.content)
            await self.retriever.add_to_faiss_index(vector, article.id)

        await self.retriever.build_faiss_index()
        query_vector = self.retriever.embed_article(text.text)
        relevant_ids = self.retriever.retrieve_relevant_articles(query_vector)

        async with self.db_session as session:
            result = await session.execute(
                select(ArticleModel).where(ArticleModel.id.in_(relevant_ids))
            )
            relevant_articles = result.scalars().all()

        user_text_with_degree = f"{text.text}\n\n[Academic Level: {user.degree}]"

        generated_text = self.generator.generate_article(
            user_text=user_text_with_degree,
            relevant_articles=relevant_articles,
        )

        async with self.db_session as session:
            new_article = AIModel(
                title=f"Generated Article for {user.degree}",
                model_output=generated_text,
                user_id=user.id,
                text_id=text.id,
                article_id=uuid4(),
                created_at=datetime.utcnow(),
                translated_output=None,
            )
            session.add(new_article)
            await session.commit()

        return {
            "success": "Article created successfully",
            "article_id": str(new_article.id),
        }

    async def list_user_generated_articles(self, user_id: UUID) -> list:
        async with self.db_session as session:
            result = await session.execute(
                select(AIModel).where(AIModel.user_id == user_id)
            )
            articles = result.scalars().all()

            if not articles:
                raise HTTPException(
                    status_code=404, detail="No generated articles found for this user."
                )

            return [
                {
                    "id": str(article.id),
                    "title": article.title,
                    "model_output": article.model_output,
                    "created_at": article.created_at,
                }
                for article in articles
            ]

    async def delete_article(self, article_id: UUID) -> None:
        async with self.db_session as session:
            article = await session.execute(
                select(AIModel).where(AIModel.id == article_id)
            )
            article = article.scalars().first()
            if article is None:
                raise HTTPException(status_code=404, detail="Text not found.")

            await session.delete(article)
            await session.commit()


class FileOperation:
    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def generate_pdf(self, article_id: UUID, user_id: UUID) -> dict:
        async with self.db_session as session:
            article = await session.execute(
                select(AIModel).where(
                    AIModel.id == article_id, AIModel.user_id == user_id
                )
            )
            article = article.scalars().first()
            if not article:
                raise HTTPException(
                    status_code=404, detail="Article not found or access denied."
                )

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(190, 10, article.model_output)

            pdf_buffer = BytesIO()
            pdf_output = pdf.output(dest="S").encode("latin-1")
            pdf_buffer = BytesIO()
            pdf_buffer.write(pdf_output)
            pdf_buffer.seek(0)

            file_name = f"article_{article.id}.pdf"
            file_path = save_file(file_name, pdf_buffer)

            new_file = FileModel(
                ai_model_id=article.id,
                user_id=user_id,
                file_name=file_name,
                file_path=file_path,
                file_type="pdf",
                created_at=datetime.utcnow(),
            )
            session.add(new_file)
            await session.commit()

            return {"download_url": f"/download/{new_file.id}"}

    async def generate_word(self, article_id: UUID, user_id: UUID) -> dict:
        async with self.db_session as session:
            article = await session.execute(
                select(AIModel).where(
                    AIModel.id == article_id, AIModel.user_id == user_id
                )
            )
            article = article.scalars().first()
            if not article:
                raise HTTPException(
                    status_code=404, detail="Article not found or access denied."
                )

            doc = Document()
            doc.add_heading(article.title, level=1)
            doc.add_paragraph(article.model_output)

            word_buffer = BytesIO()
            doc.save(word_buffer)
            word_buffer.seek(0)

            file_name = f"article_{article.id}.docx"
            file_path = save_file(file_name, word_buffer)

            new_file = FileModel(
                ai_model_id=article.id,
                user_id=user_id,
                file_name=file_name,
                file_path=file_path,
                file_type="word",
                created_at=datetime.utcnow(),
            )
            session.add(new_file)
            await session.commit()

            return {"download_url": f"/download/{new_file.id}"}

    async def get_file(self, file_id: UUID, user_id: UUID) -> dict:
        async with self.db_session as session:
            file = await session.execute(
                select(FileModel).where(
                    FileModel.id == file_id, FileModel.user_id == user_id
                )
            )
            file = file.scalars().first()
            if not file:
                raise HTTPException(
                    status_code=404, detail="File not found or access denied."
                )

            return {"file_name": file.file_name, "file_path": file.file_path}


class TranslationOperation:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def translate_article(self, article_id: UUID):
        result = await self.db_session.execute(
            select(AIModel).where(AIModel.id == article_id)
        )
        ai_article = result.scalar_one_or_none()

        if not ai_article:
            raise HTTPException(status_code=404, detail="Article not found.")

        if not ai_article.model_output:
            raise HTTPException(
                status_code=400, detail="Article has no content to translate."
            )

        original_text = ai_article.model_output
        chunk_size = 3000
        chunks = [
            original_text[i : i + chunk_size]
            for i in range(0, len(original_text), chunk_size)
        ]

        translated_chunks = []
        for chunk in chunks:
            translated = GoogleTranslator(source="auto", target="fa").translate(chunk)
            translated_chunks.append(translated)

        full_translation = "\n".join(translated_chunks)
        ai_article.translated_output = full_translation

        await self.db_session.commit()

        return {
            "detail": "Article translated and saved successfully.",
            "translated_text": full_translation,
        }
