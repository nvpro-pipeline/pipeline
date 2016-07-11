vec3 mdl_base_flakeNoiseBumpTexture( in _base_textureCoordinateInfo uvw, in float scale, in float strength, in int noiseType, in float maximumSize, in int metric, in vec3 normal )
{
  vec3 pos = uvw.position / scale;

  float cellDistance = 0.0f;
  if ( noiseType == 1 )
  {
    worleyReturn ret = worleyNoise( pos, 1.0f, metric );
    cellDistance = ret.val.x;
    pos = ret.nearest_pos_0;
  }

  vec4 ret2 = miNoise( pos );

  if ( noiseType == 0 )
  {
    pos += ret2.xyz * 2.0f;   // Displace the coordinate according to noise value

    // Then use only integer coordinates, to make flake transients "harder" and not "wobbly"
    ret2 = miNoise( ivec3( int(floor( pos.x )), int(floor( pos.y )), int(floor( pos.z ) ) ) );
  }

  float reflectivity;
  if ( ( noiseType == 1 ) && ( maximumSize < cellDistance  ) )
  {
    ret2.xyz = vec3( 0.0f, 0.0f, 0.0f );
  }

  return( normalize( normal * ( ret2.z * 1.0f / strength +1.0f ) + uvw.tangentU * ret2.x * 1.0f / strength + uvw.tangentV * ret2.y * 1.0f / strength ) );
}
