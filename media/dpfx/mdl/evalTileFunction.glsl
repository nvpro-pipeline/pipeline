int evalTileFunction( in vec2 tex, in vec2 numColsRows, in ivec2 specialColRowIndex, in vec2 specialColRowSizeFactor, in float oddRowOffset
                    , in float randomRowOffset, in float seed, in float missingTileAmount, in vec2 groutSize, in float groutRoughness
                    , in float tileBrightnessVariation, in out vec3 tileColor, bool applyColorVariance )
{
  // Pattern is constrained to uv [0,1), and just repeats outside of that
  vec2 adjustedNumColsRows = numColsRows;
  vec2 xy = ( tex - floor( tex ) ) * adjustedNumColsRows;

  // Apply additional row/column multiplier for every nth row/column
  ivec2 index = ivec2( xy );
  // We add a +1 to do more like base-1 indexing, makes more sense for user I guess
  if ( ( ( ( index.x + 1 ) % specialColRowIndex.x ) == 0 )
    && ( ( ( index.y + 1 ) % specialColRowIndex.y ) == 0 ) )
  {
    adjustedNumColsRows *= specialColRowSizeFactor;
    xy *= specialColRowSizeFactor;
    index.y = int( xy.y );
  }

  // Apply tile offset for every other row (odd positive or even negative)
  if ( ( index.y & 1 ) != 0 )
  {
    xy.x += oddRowOffset;
  }

  // Apply additional 'random' tile offset
  if ( 0.0f < randomRowOffset )
  {
    xy.x += randomRowOffset * perlinNoise( floor( xy.y ) * seed );
  }

  // Calculate the column index now that we've applied all possible modifications to x
  vec2 indexF = floor( xy );

  // Determine the relative position inside the tile (tile is defined as [0,1)
  xy -= indexF;

  // Insert 'random' holes (i.e. brick renders using grout color)
  if ( 0.0f < missingTileAmount )
  {
    float r0 = perlinNoise( indexF * seed );

    // Remapping noise samples to uniform samples in [0,1): the noise function isn't linear but
    // is symmetric so adding the inverse of negative values to positive values should provide a
    // linear sample distribution.
    if ( ( ( 0.0f <= r0 ) ? r0 : r0 + 1.0f ) < missingTileAmount )
    {
      return( 0 );
    }
  }

  // Calculate the grout size in tile-space
  vec2 adjustedGroutSize = groutSize * adjustedNumColsRows;

  // Apply a noise function to roughness to grout edges
  if ( 0.0f < groutRoughness )
  {
    // Noise frequency modifiers. Compresses the noise along the edge direction.
    float noiseScaleY = 10.0f * ( adjustedNumColsRows.y + adjustedNumColsRows.x );          // 10 is magic, adjustedNumColsRows.y+adjustedNumColsRows.x is to have noise adapt to tile size
    float noiseScaleX = ( adjustedNumColsRows.x / adjustedNumColsRows.y) * noiseScaleY;     // apply aspect ratio, to have constant noise frequency when tile is stretched

    float r1 = perlinNoise( tex * vec2( noiseScaleX, noiseScaleY ) );
    float r2 = perlinNoise( tex * vec2( noiseScaleY, noiseScaleX ) );

    // Apply factor between original and new number of row/col, to have the noise amplitude remain the same for large or small tiles.
    adjustedGroutSize.x += ( adjustedNumColsRows.x / numColsRows.x ) * groutRoughness * r1;

    // Apply aspect ratio to make the noise even in height and width and
    // Apply factor between original and new number of row/col, to have the noise amplitude remain the same
    adjustedGroutSize.y += ( square( adjustedNumColsRows.y ) / ( adjustedNumColsRows.x * numColsRows.y ) ) * groutRoughness * r2;
  }

  // Determine if we're in the tile or the grout
  if ( ( xy.x < adjustedGroutSize.x ) || ( ( 1.0f - adjustedGroutSize.x ) < xy.x )
    || ( xy.y < adjustedGroutSize.y ) || ( ( 1.0f - adjustedGroutSize.y ) < xy.y ) )
  {
    return( 0 );
  }
  else
  {
    // Tile -> Apply 'random' color variation
    if ( applyColorVariance && ( 0.0f < tileBrightnessVariation ) )
    {
      float r3 = perlinNoise( indexF * seed * 0.5f );
      float variation = tileBrightnessVariation * r3;
      tileColor = clamp( tileColor * ( 1.0f + variation ), 0.0f, 1.0f );
    }
    return( 1 );
  }
}

