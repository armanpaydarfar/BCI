"""
panel — collaborator classes extracted from the control_panel.py god class.

control_panel.py is being decomposed by composition: cohesive subsystems move
into focused collaborator objects here, and ControlPanel becomes a thin assembler
that constructs them and wires Qt widgets to them. Behaviour is preserved at the
UX level (same operator-visible behaviour, same subprocess/UDP wire), verified by
tests/test_control_panel_construction.py plus per-collaborator unit tests.

First extracted: ProcessManager (QProcess subprocess lifecycle).
"""
