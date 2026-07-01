"""
cli_formatter.py — Enterprise CLI Formatting & Styling Utility.

Contains helpers for ANSI coloring, unicode box panels, styled tables, and
custom block charts, ensuring a clean and formatted "MNC executive" look.
"""
import os
import re
import sys
from typing import List, Dict, Any, Tuple

# Enable Virtual Terminal Processing (ANSI escape sequences) in Windows Command Prompt/PowerShell
def init_terminal() -> None:
    if sys.platform == "win32":
        try:
            # os.system('') turns on ANSI escape sequence support in Windows 10+ consoles
            os.system('')
        except Exception:
            pass

# ANSI Codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
UNDERLINE = "\033[4m"

# Foreground Colors
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
GRAY = "\033[90m"

# Bright Foreground Colors
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_WHITE = "\033[97m"

# Status Indicators
OK_ICON = f"{GREEN}✔{RESET}"
WARN_ICON = f"{YELLOW}⚠{RESET}"
ERR_ICON = f"{RED}✖{RESET}"
INFO_ICON = f"{CYAN}ℹ{RESET}"

ANSI_REGEX = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')

def strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences from a string to compute its visual length."""
    return ANSI_REGEX.sub('', s)

def visual_len(s: str) -> int:
    """Return the length of the string ignoring ANSI styling codes."""
    return len(strip_ansi(s))

def style(text: str, *styles: str) -> str:
    """Apply styling codes to text and reset at the end."""
    prefix = "".join(styles)
    return f"{prefix}{text}{RESET}"

def make_bar(val: float, max_val: float, length: int = 30, color_code: str = GREEN) -> str:
    """
    Generate a clean horizontal bar using block characters.
    Uses gradient blocks for fractional endings if supported, or solid block.
    """
    if max_val <= 0:
        return "░" * length
    
    fraction = min(max(val / max_val, 0.0), 1.0)
    filled_len = fraction * length
    full_blocks = int(filled_len)
    remainder = filled_len - full_blocks
    
    # Block characters
    solid_block = "█"
    
    # Sub-block elements for smooth ends
    # 7/8, 3/4, 5/8, 1/2, 3/8, 1/4, 1/8
    blocks = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
    
    part_idx = int(remainder * 8)
    part_char = blocks[part_idx] if part_idx > 0 else ""
    
    bar_str = solid_block * full_blocks + part_char
    padding = " " * (length - len(bar_str))
    
    return f"{color_code}{bar_str}{RESET}{padding}"

def make_panel(
    title: str,
    content_lines: List[str],
    style_color: str = CYAN,
    width: int = 78,
    border_type: str = "single"
) -> str:
    """
    Draw a framed card panel around multiple lines of text.
    Handles internal coloring correctly without disrupting the box border alignment.
    """
    # Border symbols
    if border_type == "double":
        tl, tr, bl, br = "╔", "╗", "╚", "╝"
        h, v = "═", "║"
        div_l, div_r = "╠", "╣"
    else:
        tl, tr, bl, br = "┌", "┐", "└", "┘"
        h, v = "─", "│"
        div_l, div_r = "├", "┤"

    styled_v = style(v, style_color)

    # Header Border
    clean_title = strip_ansi(title)
    if clean_title:
        title_len = len(clean_title)
        # Check if title fits
        if title_len + 4 > width:
            width = title_len + 6
        
        left_h = (width - title_len - 4) // 2
        right_h = width - title_len - 4 - left_h
        
        border_top = (
            style(tl + h * left_h + "[ ", style_color)
            + title
            + style(" ]" + h * right_h + tr, style_color)
        )
    else:
        border_top = style(tl + h * (width - 2) + tr, style_color)

    border_bottom = style(bl + h * (width - 2) + br, style_color)

    # Format content lines
    formatted_lines = []
    formatted_lines.append(border_top)

    for line in content_lines:
        if line == "---":
            # Section divider line inside the panel
            formatted_lines.append(style(div_l + h * (width - 2) + div_r, style_color))
        else:
            # Normal content line
            v_len = visual_len(line)
            # Clip if line is wider than panel
            if v_len > width - 4:
                # Need to clip carefully while preserving ANSI codes (rough approximation)
                line_to_print = line[:width - 7] + "..."
                v_len = visual_len(line_to_print)
            else:
                line_to_print = line
                
            padding = " " * (width - 4 - v_len)
            formatted_lines.append(f"{styled_v}  {line_to_print}{padding}  {styled_v}")

    formatted_lines.append(border_bottom)
    return "\n".join(formatted_lines)

def make_table(
    headers: List[str],
    rows: List[List[Any]],
    alignments: List[str],  # 'left', 'right', or 'center'
    border_color: str = GRAY,
    width_padding: int = 1
) -> str:
    """
    Format a list of data rows into a clean, alignment-aware Unicode table.
    """
    # 1. Determine column widths based on maximum length of headers and values
    num_cols = len(headers)
    col_widths = [len(strip_ansi(h)) for h in headers]
    
    for row in rows:
        for i in range(min(num_cols, len(row))):
            val_str = str(row[i])
            col_widths[i] = max(col_widths[i], visual_len(val_str))
            
    # Apply padding
    col_widths = [w + 2 * width_padding for w in col_widths]
    
    # 2. Draw table borders
    tl, tr, bl, br = "┌", "┐", "└", "┘"
    h, v = "─", "│"
    tc, bc, cc = "┬", "┴", "┼"
    ml, mr = "├", "┤"
    
    styled_h = style(h, border_color)
    styled_v = style(v, border_color)
    
    # Build horizontal divider lines
    top_line = style(tl, border_color) + style(tc, border_color).join(style(h * w, border_color) for w in col_widths) + style(tr, border_color)
    mid_line = style(ml, border_color) + style(cc, border_color).join(style(h * w, border_color) for w in col_widths) + style(mr, border_color)
    bottom_line = style(bl, border_color) + style(bc, border_color).join(style(h * w, border_color) for w in col_widths) + style(br, border_color)
    
    # Helper to align cell text
    def align_cell(text: str, width: int, alignment: str) -> str:
        text_str = str(text)
        v_len = visual_len(text_str)
        fill_len = width - v_len - 2 * width_padding
        if fill_len < 0:
            fill_len = 0
            
        pad_str = " " * width_padding
        
        if alignment == "right":
            return pad_str + " " * fill_len + text_str + pad_str
        elif alignment == "center":
            left_fill = fill_len // 2
            right_fill = fill_len - left_fill
            return pad_str + " " * left_fill + text_str + " " * right_fill + pad_str
        else: # left
            return pad_str + text_str + " " * fill_len + pad_str

    # 3. Format header row
    header_cells = [align_cell(style(headers[i], BOLD), col_widths[i], alignments[i]) for i in range(num_cols)]
    header_row = styled_v + styled_v.join(header_cells) + styled_v
    
    # 4. Format data rows
    table_rows = [top_line, header_row, mid_line]
    for row in rows:
        cells = []
        for i in range(num_cols):
            val = row[i] if i < len(row) else ""
            cells.append(align_cell(str(val), col_widths[i], alignments[i]))
        table_rows.append(styled_v + styled_v.join(cells) + styled_v)
        
    table_rows.append(bottom_line)
    return "\n".join(table_rows)
