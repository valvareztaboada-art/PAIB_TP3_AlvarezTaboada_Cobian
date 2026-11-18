"""Widget del TP3. Uso: pixi run widget"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def main():
    import napari
    from tp3.ui.ventana import crear_ventana

    crear_ventana()
    napari.run()


if __name__ == "__main__":
    main()
