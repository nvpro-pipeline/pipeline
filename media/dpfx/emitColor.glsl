// emitColor is the function called at the end of the fragment shader to emit the calculated color
// There are a couple of emitColor functions (like in emitColorDepth.glsl, emitColorOITAllCounter.glsl, etc.) which reflect
// the various needs of the various algorithms.

layout(location = 0, index = 0) out vec4 Color;

// Simplest version of emitColor(), which just sets the color to the provided color.
void emitColor( in vec4 color )
{
  Color = color;
}