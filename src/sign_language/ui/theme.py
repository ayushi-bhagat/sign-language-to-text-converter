"""UI appearance theme and color conversion helpers."""

from __future__ import annotations

from sign_language.config.schema import UIConfig


def hex_to_bgr(hex_str: str) -> tuple[int, int, int]:
    """Convert hex color '#RRGGBB' to OpenCV BGR integer tuple (B, G, R)."""
    h = hex_str.lstrip("#")
    if len(h) != 6:
        return (255, 255, 255)
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


class Theme:
    """Manages application color palette for both CustomTkinter (hex) and OpenCV HUD (BGR)."""

    def __init__(self, ui_config: UIConfig) -> None:
        self.app = ui_config.appearance
        self.font_family = ui_config.font_family

        # Hex values (for CustomTkinter)
        self.accent_hex = self.app.accent
        self.success_hex = self.app.success
        self.warning_hex = self.app.warning
        self.danger_hex = self.app.danger
        self.bg_hex = self.app.bg
        self.surface_hex = self.app.surface
        self.text_primary_hex = self.app.text_primary
        self.text_muted_hex = self.app.text_muted

        # BGR values (for OpenCV drawing)
        self.accent_bgr = hex_to_bgr(self.accent_hex)
        self.success_bgr = hex_to_bgr(self.success_hex)
        self.warning_bgr = hex_to_bgr(self.warning_hex)
        self.danger_bgr = hex_to_bgr(self.danger_hex)
        self.bg_bgr = hex_to_bgr(self.bg_hex)
        self.surface_bgr = hex_to_bgr(self.surface_hex)
        self.text_primary_bgr = hex_to_bgr(self.text_primary_hex)
        self.text_muted_bgr = hex_to_bgr(self.text_muted_hex)
