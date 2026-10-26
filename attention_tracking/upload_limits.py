"""Bound multipart bodies before framework parsing, including chunked uploads."""

import asyncio
import tempfile
from starlette.responses import JSONResponse


class UploadLimits:
    def __init__(self, app, max_bytes, timeout_seconds=60):
        self.app = app
        self.max_bytes = max_bytes
        self.timeout_seconds = timeout_seconds
        self.uploading = False

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != "/jobs"
        ):
            return await self.app(scope, receive, send)
        state = scope["app"].state
        if self.uploading or getattr(state, "busy", False):
            return await JSONResponse(
                {"detail": "A video job or upload is already running"}, status_code=409
            )(scope, receive, send)
        self.uploading = True
        try:
            headers = dict(scope.get("headers", []))
            try:
                size = int(headers.get(b"content-length", b"0"))
            except ValueError:
                size = -1
            if size < 0 or size > self.max_bytes:
                return await JSONResponse(
                    {"detail": "Upload exceeds configured limit"}, status_code=413
                )(scope, receive, send)
            with tempfile.SpooledTemporaryFile(max_size=1024**2) as stream:
                total = 0
                try:
                    async with asyncio.timeout(self.timeout_seconds):
                        while True:
                            message = await receive()
                            if message["type"] == "http.disconnect":
                                return
                            body = message.get("body", b"")
                            total += len(body)
                            if total > self.max_bytes:
                                return await JSONResponse(
                                    {"detail": "Upload exceeds configured limit"},
                                    status_code=413,
                                )(scope, receive, send)
                            stream.write(body)
                            if not message.get("more_body"):
                                break
                except TimeoutError:
                    return await JSONResponse(
                        {"detail": "Upload deadline exceeded"}, status_code=408
                    )(scope, receive, send)
                stream.seek(0)

                async def replay():
                    chunk = stream.read(1024**2)
                    return {
                        "type": "http.request",
                        "body": chunk,
                        "more_body": stream.tell() < total,
                    }

                await self.app(scope, replay, send)
        finally:
            self.uploading = False
