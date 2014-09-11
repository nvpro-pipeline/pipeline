
void main(void)
{
  // Laplace filter kernel
  // 0  1  0
  // 1 -4  1
  // 0  1  0

  vec2 invSize = 1.0f / vec2( textureSize( selection, 0 ) );
  float sampleValue = -4.0f * texture( selection, varTexCoord0, 0 ).x;          // center
  sampleValue += texture( selection, vec2(varTexCoord0.x, varTexCoord0.y - invSize.y), 0 ).x; // bottom
  sampleValue += texture( selection, vec2(varTexCoord0.x - invSize.x, varTexCoord0.y), 0 ).x; // left
  sampleValue += texture( selection, vec2(varTexCoord0.x + invSize.x, varTexCoord0.y), 0 ).x; // right
  sampleValue += texture( selection, vec2(varTexCoord0.x, varTexCoord0.y + invSize.y), 0 ).x; // top

  if ( sampleValue <= 0.5f )    discard;
  Color = vec4( 1.0f, 1.0f, 0.0f, 1.0f ); // yellow
}
