"""
Minimal Avatar Pipeline Web Studio — stripped to diagnose Railway 502.
No heavy dependencies (cv2, mediapipe, numpy) at startup.
"""
import os
import logging
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebStudio")

app = FastAPI(title="Avatar Pipeline Web Studio")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok"})


@app.get("/")
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(
        "<h1>Avatar Pipeline Web Studio</h1>"
        "<p>Server is running! Static files will load once configured.</p>"
    )


# Serve static assets manually (CSS, JS, images) without StaticFiles mount
@app.get("/static/{path:path}")
async def serve_static(path: str):
    file_path = os.path.join(STATIC_DIR, path)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse(status_code=404, content="Not found")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    logger.info(f"Starting server on 0.0.0.0:{port}")
    uvicorn.run("web.app:app", host="0.0.0.0", port=port)

