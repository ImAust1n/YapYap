from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os

app = FastAPI()

@app.get("/")
async def serve_website():
    """
    Serve the website.html file.
    """
    # Use the directory of this script to find website.html
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_file_path = os.path.join(base_dir, "website.html")
    
    try:
        with open(html_file_path, "r", encoding="utf-8") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: website.html not found</h1>", status_code=404)
