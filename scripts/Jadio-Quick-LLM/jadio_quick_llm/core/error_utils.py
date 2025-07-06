"""
error_utils.py - Standardized error responses for Jadio-quick-llm API
"""
from fastapi.responses import JSONResponse
from fastapi import status
import logging
from typing import Optional, Dict

def json_error(message: str, code: int = status.HTTP_400_BAD_REQUEST, detail: Optional[str] = None, extra: Optional[Dict] = None):
    resp = {"error": message}
    if detail:
        resp["detail"] = detail
    if extra:
        resp.update(extra)
    logging.error(f"API error {code}: {message} - {detail if detail else ''}")
    return JSONResponse(status_code=code, content=resp)

def error_unauthorized(detail: Optional[str] = None):
    return json_error("Unauthorized", code=status.HTTP_401_UNAUTHORIZED, detail=detail)

def error_forbidden(detail: Optional[str] = None):
    return json_error("Forbidden", code=status.HTTP_403_FORBIDDEN, detail=detail)

def error_not_found(detail: Optional[str] = None):
    return json_error("Not found", code=status.HTTP_404_NOT_FOUND, detail=detail)

def error_server(detail: Optional[str] = None):
    return json_error("Internal server error", code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)

def error_service_unavailable(detail: Optional[str] = None):
    return json_error("Service unavailable", code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
