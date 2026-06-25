"""panel/ui_utils.py — small shared Qt helpers for the control panel and its
collaborator widgets. Leaf module (no panel imports) so any collaborator can use
these without an import cycle."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget, QSizePolicy


def _fixed_v(widget: QWidget) -> QWidget:
    """Pin a widget's vertical size policy so it stops absorbing leftover
    grid space. QWidget defaults to Preferred-vertical, which makes any
    HBox-holder row in a QGridLayout stretch to 4-5x its natural height
    when the panel has spare vertical room. Fixed clamps it at the
    sizeHint."""
    sp = widget.sizePolicy()
    sp.setVerticalPolicy(QSizePolicy.Fixed)
    widget.setSizePolicy(sp)
    return widget
