"""Agrega src/ al path para que los scripts importen tp1, tp2 y tp3."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
