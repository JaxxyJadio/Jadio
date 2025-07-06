import pkgutil
import inspect
import importlib

# List of PyQt6 modules to scan for classes
PYQT_MODULES = [
    'PyQt6.QtWidgets',
    'PyQt6.QtGui',
    'PyQt6.QtCore',
    'PyQt6.QtSvgWidgets',
    'PyQt6.QtMultimedia',
    'PyQt6.QtPrintSupport',
    'PyQt6.QtOpenGLWidgets',
]

def list_pyqt6_classes():
    all_classes = set()
    for modname in PYQT_MODULES:
        try:
            mod = importlib.import_module(modname)
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if obj.__module__.startswith('PyQt6.'):
                    all_classes.add(f'{obj.__module__}.{name}')
        except Exception as e:
            print(f'Could not import {modname}: {e}')
    return sorted(all_classes)

def main():
    print('All PyQt6 classes:')
    for cls in list_pyqt6_classes():
        print(cls)

if __name__ == '__main__':
    main()
