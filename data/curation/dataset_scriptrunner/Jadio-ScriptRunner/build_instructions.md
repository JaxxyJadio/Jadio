# Building a Standalone EXE and Installer for Code.AIgent

This guide explains how to package Code.AIgent as a standalone Windows executable and create an installer for GitHub releases.

## 1. Requirements
- Windows 10/11
- Python 3.9+
- pip
- [PyInstaller](https://pyinstaller.org/) (`pip install pyinstaller`)
- (Optional) [NSIS](https://nsis.sourceforge.io/) for installer creation

## 2. Prepare the Environment
1. Ensure all dependencies are installed:
   ```sh
   pip install -r requirements.txt
   pip install pyinstaller
   ```
2. Clean previous builds:
   ```sh
   rmdir /s /q dist build
   del Code.AIgent.spec
   ```

## 3. Build the Standalone EXE
1. From the project root, run:
   ```sh
   pyinstaller pyinstaller.spec
   ```
   - This creates `dist/Code.AIgent/Code.AIgent.exe`.
   - You can copy/rename it to `Code.AIgent.exe` for release.

2. Test the EXE:
   ```sh
   dist\Code.AIgent\Code.AIgent.exe
   ```

## 4. Create an Installer (Recommended for GitHub Releases)
1. Install [NSIS](https://nsis.sourceforge.io/).
2. Use the provided `installer.nsi` file (edit as needed for your files/paths).
3. Compile the installer:
   - Open NSIS, select `installer.nsi`, and click "Compile".
   - The installer EXE will be created in the project directory.

## 5. Upload to GitHub Releases
- Attach both the standalone EXE and the installer EXE to your GitHub release.
- Include a README note for users: "If you see a Windows Defender SmartScreen warning, click 'More info' > 'Run anyway'."

---
For advanced options (icon, version info, etc.), see the PyInstaller and NSIS documentation.
