_base_textureReturn mdl_base_flakeNoiseTexture( in _base_textureCoordinateInfo uvw, in float intensity, in float scale, in float density, int noiseType, float maximumSize, int metric )
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
  float scal = ret2.w;

  if ( noiseType == 0 )
  {
    pos += ret2.xyz * 2.0f;   // Displace the coordinate according to noise value

    // Then use only integer coordinates, to make flake transients "harder" and not "wobbly"
    scal = miNoise( ivec3( int(floor( pos.x )), int(floor( pos.y )), int(floor( pos.z ) ) ) ).w;
  }

  float reflectivity;
  if ( ( noiseType == 1 ) && ( maximumSize < cellDistance  ) )
  {
    reflectivity = 0.0f;
  }
  else
  {
    reflectivity = pow( scal, 1.0f / density ) * intensity;
  }

  _base_textureReturn tr;
  tr.tint = vec3( reflectivity, reflectivity, reflectivity );
  tr.mono = reflectivity;
  return( tr );
}
