_base_textureCoordinateInfo mdl_base_textureCoordinateInfo( in vec3 pos, in vec3 tangentU, in vec3 tangentV )
{
  _base_textureCoordinateInfo tci;
  tci.position  = pos;
  tci.tangentU = normalize( tangentU );
  tci.tangentV = normalize( tangentV );
  return( tci );
}

