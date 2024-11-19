from pydantic import BaseModel, Field
from typing import List
from pydantic import validator


class QASchedulerConfig(BaseModel):
    n_qa_per_context: int = Field(
        ..., description="Number of QA pairs to generate per context"
    )
    max_items: int = Field(..., description="Maximum number of items to generate")


class ConversationSchedulerConfig(BaseModel):
    n_new_conversations: int = Field(
        ..., description="Number of new conversations to generate"
    )
    n_previous_conversations: int = Field(
        ..., description="Number of previous conversations to keep from the dataset"
    )
    max_items: int = Field(..., description="Maximum number of items to generate")


class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")


class QASet(BaseModel):
    questions: List[str] = Field(..., description="List of generated questions")
    answers: List[str] = Field(..., description="List of corresponding answers")
    total_chars: int = Field(
        ..., description="Total character count of questions and answers"
    )
    context_seed: str = Field(
        ..., description="Original context used to generate QA pairs"
    )

    @validator("questions", "answers")
    def validate_questions_and_answers(cls, v):
        if not v:
            raise ValueError("Questions and answers must be non-empty lists")
        return v


class Conversation(BaseModel):
    messages: List[Message] = Field(..., description="List of conversation messages")
    total_chars: int = Field(
        ..., description="Total character count of the conversation"
    )
    messages_seed: List[Message] = Field(
        ..., description="Original messages used to generate conversation"
    )

    @validator("messages")
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages must be non-empty lists")
        return v
