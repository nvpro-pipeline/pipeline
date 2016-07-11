_base_textureCoordinateInfo mdl_base_transformCoordinate( in mat4 transform, in _base_textureCoordinateInfo coordinate )
{
  _base_textureCoordinateInfo tci;
  tci.position = ( transform * vec4( coordinate.position, 1.0f ) ).xyz;
  tci.tangentU = ( transform * vec4( coordinate.tangentU, 0.0f ) ).xyz;
  tci.tangentV = ( transform * vec4( coordinate.tangentV, 0.0f ) ).xyz;
  return( tci );
}

