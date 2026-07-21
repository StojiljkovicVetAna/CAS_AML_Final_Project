"""Run the backend with ``python -m final_project.backend``."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "final_project.backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
