_base_textureReturn mdl_base_checkerTexture( in _base_textureCoordinateInfo uvw, in vec3 color1, in vec3 color2, in float blur, in float checkerPosition )
{
  // Get the fractional uv
  vec3 tex = uvw.position - floor( uvw.position );
  vec3 relTex = tex - checkerPosition;

  // Determine what part of the checker we're in
  ivec3 intTex = floatBitsToInt( relTex );
  bool inColOne = ( ( intTex.x ^ intTex.y ^ intTex.z ) < 0 );

  // Calculate distance to the closest edge in each dimension
  vec3 edgeDist = min( tex, min( 1.0f - tex, abs( relTex ) ) );

  // Calculate the amount of blending for each dimension, where 1.0 means no blending and 0.0 means full blending.
  // Total amout of blending by combining the blends from all dimensions to get smoother corners.
  // Scaled to [0.0, 0.5].
  float blendAmount = ( edgeDist.x < blur ) ? edgeDist.x / blur : 1.0f;
  if ( edgeDist.y < blur )
  {
    blendAmount *= edgeDist.y / blur;
  }
  // for 2D textures, edgeDist.z is always zero, and thus would set blendAmount to zero
  if ( edgeDist.z < blur )
  {
    blendAmount *= edgeDist.z / blur;
  }
  blendAmount *= 0.5f;

  _base_textureReturn tr;
  tr.tint = mix( color2, color1, inColOne ? 0.5f + blendAmount : 0.5f - blendAmount );
  tr.mono = 0;
  return( tr );
}

