_base_textureReturn mdl_base_tileTexture( in _base_textureCoordinateInfo uvw, in vec3 tileColor, in vec3 groutColor, in float numberOfRows
                                        , in float numberOfColumns, in float groutWidth, in float groutHeight, in float groutRoughness
                                        , in float missingTileAmount, in float tileBrightnessVariation, in float seed, in int specialRowIndex
                                        , in float specialRowWidthFactor, in int specialColumnIndex, in float specialColumnHeightFactor
                                        , in float oddRowOffset, in float randomRowOffset )
{
  vec2 numColsRows = vec2( numberOfColumns, numberOfRows );
  ivec2 specialColRowIndex = ivec2( specialColumnIndex, specialRowIndex );
  vec2 specialColRowSizeFactor = vec2( specialColumnHeightFactor, specialRowWidthFactor );
  vec2 groutSize = vec2( groutWidth, groutHeight );
  int r = evalTileFunction( uvw.position.xy, numColsRows, specialColRowIndex, specialColRowSizeFactor, oddRowOffset, randomRowOffset
                          , seed, missingTileAmount, groutSize, groutRoughness, tileBrightnessVariation, tileColor, true );

  _base_textureReturn tr;
  tr.tint = ( r != 0 ) ? tileColor : groutColor;
  tr.mono = 0.0f;
  return( tr );
}

