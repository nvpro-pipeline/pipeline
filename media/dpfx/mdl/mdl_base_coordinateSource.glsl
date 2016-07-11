_base_textureCoordinateInfo mdl_base_coordinateSource( in int coordinateSystem, in int textureSpace )
{
  _base_textureCoordinateInfo tci;
  switch( coordinateSystem )
  {
    case texture_coordinate_object :
      tci.position = varObjPos;
      tci.tangentU = normalize( varObjTangent );
      tci.tangentV = normalize( varObjBinormal );
      break;
    case texture_coordinate_world :
      tci.position = varWorldPos;
      tci.tangentU = normalize( varTangent );
      tci.tangentV = normalize( varBinormal );
      break;
    case texture_coordinate_uvw :
    default :
      tci.position = varTexCoord0;
      tci.tangentU = normalize( varTangent );
      tci.tangentV = normalize( varBinormal );
      break;
  }
  return( tci );
}

