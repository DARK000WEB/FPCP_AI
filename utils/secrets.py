from passlib.context import CryptContext

password_manager = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return password_manager.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_manager.verify(plain_password, hashed_password)
