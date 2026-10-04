# Project Cleanup Log

| File/Folder | Purpose | Used by | Keep/Delete | Reason |
| --- | --- | --- | --- | --- |
| `__pycache__` | Python bytecode cache | None | Delete | Standard cleanup |
| `debug/` | Temporary debug scripts | None | Delete | Unnecessary |
| `logs/` | Old training logs | None | Delete | Stale/Obsolete |
| `plots/` | Old generated plots | None | Delete | Not part of production app |
| `test_mri.png` | Manual script testing image | None | Delete | Not needed anymore |
| `dataset/` | Image dataset storage | Model/Inference | Keep | Required for local usage/testing |
| `src/` | Modular code package | app.py/predict.py | Keep | Contains entire pipeline logic |
| `app.py` | Main Streamlit App | User | Keep | Core application |
| `config.yaml` | Application configs | app.py | Keep | Configuration variables |
| `models/` | Trained weights storage | app.py | Keep | Required for inference |
| `requirements.txt` | Package dependencies | Docker/Setup | Keep | Environment build |
