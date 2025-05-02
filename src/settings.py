from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Settings for the application.
    """
    GROQ_API_KEY: str
    QDRANT_URL:str
    QDRANT_API_KEY:str
    SERPER_API_KEY:str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

settings = Settings()
