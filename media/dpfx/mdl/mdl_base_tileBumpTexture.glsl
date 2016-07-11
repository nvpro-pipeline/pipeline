vec3 mdl_base_tileBumpTexture( in _base_textureCoordinateInfo uvw, in float factor, in float numberOfRows, in float numberOfColumns, in float groutWidth
                             , in float groutHeight, in float groutRoughness, in float missingTileAmount, in float tileBrightnessVariation, in float seed
                             , in int specialRowIndex, in float specialRowWidthFactor, in int specialColumnIndex, in float specialColumnHeightFactor
                             , in float oddRowOffset, in float randomRowOffset, in vec3 normal )
{
  float delta = 0.005f;   // magic, looks good with this value

  // We sample the tile function at three positions to get a differential.
  vec3 tileVariationColor = vec3( 1.0f, 1.0f, 1.0f );

  vec2 numColsRows = vec2( numberOfColumns, numberOfRows );
  ivec2 specialColRowIndex = ivec2( specialColumnIndex, specialRowIndex );
  vec2 specialColRowSizeFactor = vec2( specialColumnHeightFactor, specialRowWidthFactor );
  vec2 groutSize = vec2( groutWidth, groutHeight );

  vec2 pos[3];
  pos[0] = uvw.position.xy;
  pos[1] = uvw.position.xy + delta * vec2( uvw.tangentV.x, -uvw.tangentV.y );
  pos[2] = uvw.position.xy + delta * vec2( uvw.tangentU.x, -uvw.tangentU.y );

  int r[3];
  for ( int i=0 ; i<3 ; i++ )
  {
    r[i] = evalTileFunction( pos[i], numColsRows, specialColRowIndex, specialColRowSizeFactor, oddRowOffset, randomRowOffset
                           , seed, missingTileAmount, groutSize, groutRoughness, tileBrightnessVariation, tileVariationColor
                           , (i == 0) ? true : ((i == 1) ? (r[0] == 0) : ((r[0] | r[1]) == 0)) );
  }

  // In 3ds max, the difference of both colors scales the bump effect.
  // We do it differently (the scale can be controlled by the user instead).
  if ( ( r[2] == r[0] ) && ( r[1] == r[0] ) )
  {
    return( normal );
  }
  else
  {
    // At the edge of a tile
    // If both colors are mapped, we don't apply edge bump (useful to create tile-ish bump maps
    // without tile relief). Otherwise, the scale depends on the difference between the non-textured colors.
    // The color variation was applied to this color which we use to scale the bump effect
    // (allowing this feature to be a "depth variation" for bricks of different depths)
    return( normalize( normal - factor * abs( mdl_math_luminance( tileVariationColor ) ) * ( ( r[1] - r[0] ) * uvw.tangentU + ( r[2] - r[0] ) * uvw.tangentV ) ) );
  }
}

