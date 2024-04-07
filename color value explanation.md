In this project, I use matplotlib.cm(jet) to project the 2-d heatmap to 3-d color map. Each level of pixel magnitude in 2-d heatmap corresponds to one color (3-channel) combinatioin in 3-d color map.

Low Values: The colormap starts with dark blue for the lowest values, indicating the "cold" part of the data spectrum.
Intermediate Low Values: As values increase, the color shifts to light blue and then to cyan, marking a transition towards the "warmer" parts of the spectrum.
Middle Values: Green colors represent the middle range of the data values. This central point serves as a midpoint in the data range.
Intermediate High Values: Further increases in value lead to a shift from green to yellow and then to orange, signaling that the values are getting "hotter."
High Values: The highest values are represented by red, with dark red indicating the very top of the value range.
