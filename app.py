from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import neuroglancer
import tifffile
import numpy as np
import os
import shutil
from contextlib import asynccontextmanager

# Global variables to hold the viewer state
GLOBAL_VIEWER = None
GLOBAL_VIEWER_URL = ""

def update_neuroglancer_layers(raw_path, pred_path):
    """Tells the Chef to clear the table and cook the new files."""
    global GLOBAL_VIEWER
    
    print(f"🔄 Updating 3D Viewer with new files...")
    
    try:
        raw_data = tifffile.imread(raw_path)
        pred_data = tifffile.imread(pred_path)
        binary_pred = (pred_data > 0.5).astype(np.uint32) 
    except Exception as e:
        print(f"❌ Error reading files: {e}")
        return

    dimensions = neuroglancer.CoordinateSpace(names=['z', 'y', 'x'], units='nm', scales=[40, 4, 4])

    # Update the live viewer
    with GLOBAL_VIEWER.txn() as s:
        s.layers.clear() # Wipe the old data
        
        s.layers['Raw Brain Tissue'] = neuroglancer.ImageLayer(
            source=neuroglancer.LocalVolume(data=raw_data, dimensions=dimensions)
        )
        s.layers['AI Cell Walls'] = neuroglancer.SegmentationLayer(
            source=neuroglancer.LocalVolume(data=binary_pred, dimensions=dimensions)
        )
    print("✅ 3D Viewer Updated Successfully!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_VIEWER, GLOBAL_VIEWER_URL
    print("🍳 Starting the Neuroglancer Kitchen...")
    
    neuroglancer.set_server_bind_address('127.0.0.1', bind_port=0)
    GLOBAL_VIEWER = neuroglancer.Viewer()
    GLOBAL_VIEWER_URL = GLOBAL_VIEWER.get_viewer_url()
    
    # Load defaults if they exist
    default_raw = "artifacts/test-input.tif"
    default_pred = "artifacts/final_submission.tif"
    if os.path.exists(default_raw) and os.path.exists(default_pred):
        update_neuroglancer_layers(default_raw, default_pred)
        
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend")

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "neuroglancer_url": GLOBAL_VIEWER_URL})

# THE NEW UPLOAD RECEIVER
@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(request: Request, raw_file: UploadFile = File(...), pred_file: UploadFile = File(...)):
    print("📥 Receiving new files from user...")
    
    # Define where to temporarily save the uploaded files
    os.makedirs("artifacts/uploads", exist_ok=True)
    raw_path = f"artifacts/uploads/{raw_file.filename}"
    pred_path = f"artifacts/uploads/{pred_file.filename}"
    
    # Save the files to the hard drive
    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(raw_file.file, buffer)
    with open(pred_path, "wb") as buffer:
        shutil.copyfileobj(pred_file.file, buffer)
        
    # Tell Neuroglancer to update
    update_neuroglancer_layers(raw_path, pred_path)
    
    # Refresh the webpage so the user can see the new render
    return templates.TemplateResponse("index.html", {"request": request, "neuroglancer_url": GLOBAL_VIEWER_URL})

if __name__ == "__main__":
    import uvicorn
    print("🚀 Booting up the Web Application...")
    uvicorn.run(app, host="127.0.0.1", port=8000)