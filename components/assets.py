from fasthtml.common import *


from fasthtml.common import *


def arrow_circle_icon():
    return (
        Svg(
            NotStr(
                """<circle cx="12" cy="12" r="10"/><path d="M8 12h8"/><path d="m12 16 4-4-4-4"/>"""
            ),
            width="24",
            height="24",
            viewBox="0 0 24 24",
            fill="none",
            stroke="currentColor",
            stroke_width="1.5px",
            cls="text-red-500"  # Red color for stroke
        ),
    )


def send_icon():
    return (
        Svg(
            NotStr("""<path d="m3 3 3 9-3 9 19-9Z"/><path d="M6 12h16"/>"""),
            width="30",
            height="30",
            viewBox="0 0 24 24",
            fill="none",
            stroke="currentColor",
            stroke_linecap="round",
            stroke_linejoin="round",
            stroke_width="1.5px",
            cls="text-red-500"  # Red color for stroke
        ),
    )




