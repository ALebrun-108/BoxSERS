"""
Author : Alexis Lebrun (PhD student)

School : Université Laval (Qc, Canada)

This module provides functions necessary for internal use, but not relevant for users
using the package
"""


def _lightdark_switch(darktheme):
    """
    Adjust some parameters of the graphs according to the selected theme (dark or light).

    Parameters:
        darktheme : boolean, default=False
            If True, returns the values associated with the dark background
            and otherwise those associated with a light background

    Returns:
        (string) Frame color. Color for the text and the plot frame.

        (string) Background color. Color for the figure, legend and plot facecolor.

        (float) Grid alpha value. The grid is slightly paled for the dark theme
    """
    if darktheme:
        return 'w', 'k', 0.3  # the grid is slightly paled for the dark theme
    return 'k', 'w', 0.4


if __name__ == "__main__":
    help(__name__)
