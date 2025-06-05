from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class FileLink(_message.Message):
    __slots__ = ("project_id", "file_id")
    PROJECT_ID_FIELD_NUMBER: _ClassVar[int]
    FILE_ID_FIELD_NUMBER: _ClassVar[int]
    project_id: str
    file_id: str
    def __init__(self, project_id: _Optional[str] = ..., file_id: _Optional[str] = ...) -> None: ...

class SignedURL(_message.Message):
    __slots__ = ("url",)
    URL_FIELD_NUMBER: _ClassVar[int]
    url: str
    def __init__(self, url: _Optional[str] = ...) -> None: ...
