
_base_textureReturn mdl_base_fileTexture( in texture2D tex, in vec3 colorOffset, in vec3 colorScale, in int monoSource, in _base_textureCoordinateInfo uvw
                                        , in vec2 cropU, in vec2 cropV, in int wrapU, in int wrapV, in bool clip )
{
  _base_textureReturn tr;
  if ( clip && ( ( ( wrapU == wrap_clamp ) && ( ( uvw.position.x < 0.0f ) || ( 1.0f < uvw.position.x ) ) )
              || ( ( wrapV == wrap_clamp ) && ( ( uvw.position.y < 0.0f ) || ( 1.0f < uvw.position.y ) ) ) ) )
  {
    tr.tint = vec3( 0.0f, 0.0f, 0.0f );
    tr.mono = 0.0f;
  }
  else
  {
    vec4 t4 = texture( tex.sampler, uvw.position.xy );
    tr.tint = pow( t4.rgb, vec3( tex.gamma ) ) * colorScale + colorOffset;
    tr.mono = monoChannel( vec4( tr.tint, t4.w ), monoSource );
  }
  return( tr );
}
