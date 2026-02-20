#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

APP_NAME="npyz-viewer"
APP_ID="local.npyz.viewer"
APP_BUNDLE="${HOME}/Applications/${APP_NAME}.app"
OLD_APP_BUNDLE="${HOME}/Applications/NPZ NPY Viewer.app"
CONTENTS_DIR="${APP_BUNDLE}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RES_DIR="${CONTENTS_DIR}/Resources"
PLIST="${CONTENTS_DIR}/Info.plist"

BIN_SRC="${PROJECT_DIR}/target/release/npyz-viewer"
BIN_DST="${MACOS_DIR}/npyz-viewer"
ICON_SRC="${PROJECT_DIR}/assets/npyz-viewer-logo.svg"
ICON_DST="${RES_DIR}/npyz-viewer-logo.svg"

echo "[1/4] Building release binary..."
cargo build --release --bin npyz-viewer --manifest-path "${PROJECT_DIR}/Cargo.toml"

echo "[2/4] Replacing old app bundle and creating: ${APP_BUNDLE}"
rm -rf "${OLD_APP_BUNDLE}"
install -d "${MACOS_DIR}" "${RES_DIR}"
install -m755 "${BIN_SRC}" "${BIN_DST}"
install -m644 "${ICON_SRC}" "${ICON_DST}"

cat >"${PLIST}" <<EOF2
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>${APP_NAME}</string>
  <key>CFBundleDisplayName</key>
  <string>${APP_NAME}</string>
  <key>CFBundleIdentifier</key>
  <string>${APP_ID}</string>
  <key>CFBundleVersion</key>
  <string>1.0</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundleExecutable</key>
  <string>npyz-viewer</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>LSApplicationCategoryType</key>
  <string>public.app-category.utilities</string>
  <key>CFBundleDocumentTypes</key>
  <array>
    <dict>
      <key>CFBundleTypeName</key>
      <string>NumPy Array</string>
      <key>CFBundleTypeExtensions</key>
      <array><string>npy</string></array>
      <key>CFBundleTypeRole</key>
      <string>Viewer</string>
      <key>LSHandlerRank</key>
      <string>Owner</string>
    </dict>
    <dict>
      <key>CFBundleTypeName</key>
      <string>NumPy Zip</string>
      <key>CFBundleTypeExtensions</key>
      <array><string>npz</string></array>
      <key>CFBundleTypeRole</key>
      <string>Viewer</string>
      <key>LSHandlerRank</key>
      <string>Owner</string>
    </dict>
  </array>
</dict>
</plist>
EOF2

echo "[3/4] Registering app in LaunchServices..."
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
if [[ -x "${LSREGISTER}" ]]; then
  "${LSREGISTER}" -f "${APP_BUNDLE}" || true
fi

echo "[4/4] Optional default-association with duti..."
if command -v duti >/dev/null 2>&1; then
  duti -s "${APP_ID}" .npy all || true
  duti -s "${APP_ID}" .npz all || true
  echo "duti applied (.npy/.npz)."
else
  echo "duti not found. If needed, set default once in Finder with 'Get Info -> Open with -> Change All'."
fi

echo
echo "Done."
echo "Test: open -a \"${APP_NAME}\" /path/to/file.npy"
