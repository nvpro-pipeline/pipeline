// emitColor is the function called at the end of the fragment shader to emit the calculated color
// There are a couple of emitColor functions (like in emitColorDepth.glsl, emitColorOITAllCounter.glsl, etc.) which reflect
// the various needs of the various algorithms.

layout(location = 0, index = 0) out vec4 Color;

// Version of emitColor() used for depth pass shaders, where the color just have to be set to something, but the actual
// value doesn't matter
void emitColor( in vec4 color )
{
  Color = vec4( 0 );
}