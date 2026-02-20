#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OLD_APP_ID="npz_npy_viewer"
APP_ID="npyz-viewer"
APP_NAME="npyz-viewer"
BIN_SRC="${PROJECT_DIR}/target/release/npyz-viewer"
BIN_DST="${HOME}/.local/bin/npyz-viewer"
DESKTOP_FILE="${HOME}/.local/share/applications/${APP_ID}.desktop"
MIME_XML="${HOME}/.local/share/mime/packages/${APP_ID}.xml"
ICON_SRC="${PROJECT_DIR}/assets/npyz-viewer-logo.svg"
ICON_DIR="${HOME}/.local/share/icons/hicolor/scalable/apps"
ICON_DST="${ICON_DIR}/${APP_ID}.svg"
OLD_BIN_DST="${HOME}/.local/bin/${OLD_APP_ID}"
OLD_DESKTOP_FILE="${HOME}/.local/share/applications/${OLD_APP_ID}.desktop"
OLD_MIME_XML="${HOME}/.local/share/mime/packages/${OLD_APP_ID}.xml"

echo "[1/6] Building release binary..."
cargo build --release --bin npyz-viewer --manifest-path "${PROJECT_DIR}/Cargo.toml"

echo "[2/6] Replacing old install and installing binary to ${BIN_DST} ..."
rm -f "${OLD_BIN_DST}" "${OLD_DESKTOP_FILE}" "${OLD_MIME_XML}"
install -Dm755 "${BIN_SRC}" "${BIN_DST}"

echo "[3/6] Installing app icon ..."
install -d "${ICON_DIR}"
install -m644 "${ICON_SRC}" "${ICON_DST}"

echo "[4/6] Installing desktop entry ..."
install -d "${HOME}/.local/share/applications"
cat >"${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Name=${APP_NAME}
Comment=Open NumPy .npy and .npz files
Exec=${BIN_DST} %f
Icon=${APP_ID}
Terminal=false
Categories=Utility;Science;
MimeType=application/x-npy;application/x-npz;
StartupNotify=true
EOF

echo "[5/6] Installing MIME definitions ..."
install -d "${HOME}/.local/share/mime/packages"
cat >"${MIME_XML}" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
  <mime-type type="application/x-npy">
    <comment>NumPy array file</comment>
    <glob pattern="*.npy"/>
  </mime-type>
  <mime-type type="application/x-npz">
    <comment>NumPy zip archive</comment>
    <glob pattern="*.npz"/>
  </mime-type>
</mime-info>
EOF

if command -v update-mime-database >/dev/null 2>&1; then
  update-mime-database "${HOME}/.local/share/mime"
else
  echo "WARN: update-mime-database not found. Install 'shared-mime-info'."
fi

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "${HOME}/.local/share/applications" || true
fi

echo "[6/6] Setting defaults for .npy/.npz ..."
if command -v xdg-mime >/dev/null 2>&1; then
  xdg-mime default "${APP_ID}.desktop" application/x-npy || true
  xdg-mime default "${APP_ID}.desktop" application/x-npz || true
else
  echo "WARN: xdg-mime not found."
fi

echo
echo "Done."
echo "Try: xdg-open /path/to/file.npy"
